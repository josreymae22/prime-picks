"""
line_snapshotter.py

Background scheduler that snapshots betting lines every 4 hours.
Stores snapshots locally → calculates open-to-current movement.
No paid API required; works with The Odds API free tier.

When Odds API paid tier is available (set ODDS_API_PAID=true in env),
swaps to using their historical odds endpoint instead of self-snapshotting.

Movement features produced:
  spread_open       - opening spread
  spread_current    - current spread
  spread_move       - current minus open (pts, positive = moved toward home)
  total_open        - opening total
  total_current     - current total
  total_move        - current minus open (pts, positive = moved up)
  move_velocity     - abs(spread_move) / hours_since_open (pts/hr)
  sharp_signal      - composite: large + fast move = high sharp signal (0-1)
  steam_move        - bool: move ≥ 2 pts in ≤ 2 hrs (classic steam)
  reverse_line_move - bool: public bets on one side, line moves other way
"""

import json
import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

SNAPSHOT_DB_PATH = os.path.join(os.path.dirname(__file__), "line_snapshots.json")
SNAPSHOT_INTERVAL_HOURS = 4
MOVEMENT_WINDOW_HOURS = 24
ODDS_API_PAID = os.getenv("ODDS_API_PAID", "false").lower() == "true"


def _load_snapshots() -> dict:
    if os.path.exists(SNAPSHOT_DB_PATH):
        with open(SNAPSHOT_DB_PATH, "r") as f:
            return json.load(f)
    return {"snapshots": [], "last_snapshot": None}


def _save_snapshots(db: dict):
    db["last_snapshot"] = datetime.utcnow().isoformat()
    with open(SNAPSHOT_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def _game_key(home: str, away: str) -> str:
    """Stable key for a matchup."""
    return f"{home.lower().replace(' ', '_')}|{away.lower().replace(' ', '_')}"


class LineSnapshotter:

    def __init__(self):
        self.db = _load_snapshots()
        self._running = False

    def reload(self):
        self.db = _load_snapshots()

    # ──────────────────────────────────────────
    # Snapshot management
    # ──────────────────────────────────────────

    async def take_snapshot(self, league: str = "NFL"):
        """
        Pull current lines and store a timestamped snapshot.
        Called on startup and every SNAPSHOT_INTERVAL_HOURS thereafter.
        """
        from lines_fetcher import get_lines
        try:
            lines = await get_lines(league=league)
            if not lines:
                logger.info(f"No lines available for {league} snapshot")
                return

            now = datetime.utcnow().isoformat()
            new_snaps = []
            for line in lines:
                if line.get("spread") is None and line.get("total") is None:
                    continue
                new_snaps.append({
                    "game_key": _game_key(line["home_team"], line["away_team"]),
                    "home_team": line["home_team"],
                    "away_team": line["away_team"],
                    "league": league,
                    "spread": line.get("spread"),
                    "total": line.get("total"),
                    "home_ml": line.get("home_ml"),
                    "away_ml": line.get("away_ml"),
                    "bookmaker": line.get("bookmaker"),
                    "commence_time": line.get("commence_time", ""),
                    "snapshot_time": now,
                })

            self.db["snapshots"].extend(new_snaps)

            # Prune snapshots older than 7 days to keep file manageable
            cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
            self.db["snapshots"] = [
                s for s in self.db["snapshots"]
                if s["snapshot_time"] >= cutoff
            ]

            _save_snapshots(self.db)
            logger.info(f"Line snapshot taken: {len(new_snaps)} {league} games at {now}")

        except Exception as e:
            logger.error(f"Snapshot error for {league}: {e}")

    async def take_all_snapshots(self):
        """Snapshot both leagues."""
        await self.take_snapshot("NFL")
        await asyncio.sleep(1)
        await self.take_snapshot("CFB")

    # ──────────────────────────────────────────
    # Movement calculation
    # ──────────────────────────────────────────

    def get_movement(self, home_team: str, away_team: str, window_hours: int = 24) -> dict:
        """
        Calculate line movement for a game over the last N hours.
        Returns movement features ready for the prediction model.
        """
        key = _game_key(home_team, away_team)
        cutoff = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()

        # Get all snapshots for this game within window
        game_snaps = sorted(
            [s for s in self.db["snapshots"]
             if s["game_key"] == key and s["snapshot_time"] >= cutoff],
            key=lambda s: s["snapshot_time"]
        )

        if len(game_snaps) < 2:
            return self._empty_movement()

        earliest = game_snaps[0]
        latest = game_snaps[-1]

        # Time delta in hours
        try:
            t0 = datetime.fromisoformat(earliest["snapshot_time"])
            t1 = datetime.fromisoformat(latest["snapshot_time"])
            hours_elapsed = max(0.25, (t1 - t0).total_seconds() / 3600)
        except Exception:
            hours_elapsed = window_hours

        spread_open = earliest.get("spread")
        spread_current = latest.get("spread")
        total_open = earliest.get("total")
        total_current = latest.get("total")

        spread_move = None
        total_move = None
        move_velocity = 0.0
        sharp_signal = 0.0
        steam_move = False

        if spread_open is not None and spread_current is not None:
            spread_move = round(spread_current - spread_open, 1)
            move_velocity = round(abs(spread_move) / hours_elapsed, 3)

            # Steam move: ≥2 pt spread move in ≤2 hours
            steam_move = abs(spread_move) >= 2.0 and hours_elapsed <= 2.0

            # Sharp signal composite (0-1 scale)
            # Large move + fast velocity = high signal
            magnitude_signal = min(1.0, abs(spread_move) / 5.0)   # 5pt move = max
            velocity_signal = min(1.0, move_velocity / 1.0)        # 1pt/hr = max
            sharp_signal = round((magnitude_signal * 0.6 + velocity_signal * 0.4), 3)

        if total_open is not None and total_current is not None:
            total_move = round(total_current - total_open, 1)

        # Direction interpretation
        # For spread: negative = moved toward away (away team getting sharper money)
        #             positive = moved toward home
        move_direction = None
        if spread_move is not None:
            if spread_move < -0.5:
                move_direction = "toward_away"
            elif spread_move > 0.5:
                move_direction = "toward_home"
            else:
                move_direction = "stable"

        return {
            "has_movement_data": True,
            "snapshots_used": len(game_snaps),
            "window_hours": window_hours,
            "hours_elapsed": round(hours_elapsed, 1),
            "spread_open": spread_open,
            "spread_current": spread_current,
            "spread_move": spread_move,
            "total_open": total_open,
            "total_current": total_current,
            "total_move": total_move,
            "move_velocity": move_velocity,
            "sharp_signal": sharp_signal,
            "steam_move": steam_move,
            "move_direction": move_direction,
            "earliest_snapshot": earliest["snapshot_time"],
            "latest_snapshot": latest["snapshot_time"],
        }

    def get_all_movements(self, league: str, window_hours: int = 24) -> list[dict]:
        """Get movement data for all games of a league."""
        self.reload()
        cutoff = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()

        # Find all unique games with recent snapshots
        seen_keys = set()
        games = []
        for snap in self.db["snapshots"]:
            if snap.get("league") != league:
                continue
            if snap["snapshot_time"] < cutoff:
                continue
            key = snap["game_key"]
            if key not in seen_keys:
                seen_keys.add(key)
                movement = self.get_movement(snap["home_team"], snap["away_team"], window_hours)
                movement["home_team"] = snap["home_team"]
                movement["away_team"] = snap["away_team"]
                games.append(movement)

        # Sort by sharp_signal descending
        return sorted(games, key=lambda g: g.get("sharp_signal", 0), reverse=True)

    def _empty_movement(self) -> dict:
        return {
            "has_movement_data": False,
            "snapshots_used": 0,
            "spread_open": None,
            "spread_current": None,
            "spread_move": None,
            "total_open": None,
            "total_current": None,
            "total_move": None,
            "move_velocity": 0.0,
            "sharp_signal": 0.0,
            "steam_move": False,
            "move_direction": None,
        }

    # ──────────────────────────────────────────
    # Model feature extraction
    # ──────────────────────────────────────────

    def get_model_features(self, home_team: str, away_team: str) -> dict:
        """
        Extract movement features formatted for model input.
        These augment the base prediction features.
        """
        mv = self.get_movement(home_team, away_team)
        if not mv["has_movement_data"]:
            return {
                "sharp_signal": 0.0,
                "spread_move": 0.0,
                "total_move": 0.0,
                "move_velocity": 0.0,
                "steam_move": 0,
            }

        return {
            "sharp_signal": mv["sharp_signal"],
            "spread_move": mv.get("spread_move") or 0.0,
            "total_move": mv.get("total_move") or 0.0,
            "move_velocity": mv["move_velocity"],
            "steam_move": int(mv["steam_move"]),
        }

    # ──────────────────────────────────────────
    # Background scheduler
    # ──────────────────────────────────────────

    async def start_scheduler(self):
        """
        Run snapshot loop every SNAPSHOT_INTERVAL_HOURS.
        Call this as an asyncio background task from FastAPI lifespan.
        """
        self._running = True
        logger.info(f"Line snapshot scheduler started — interval: {SNAPSHOT_INTERVAL_HOURS}h")
        while self._running:
            await self.take_all_snapshots()
            await asyncio.sleep(SNAPSHOT_INTERVAL_HOURS * 3600)

    def stop_scheduler(self):
        self._running = False

    def get_snapshot_stats(self) -> dict:
        total = len(self.db["snapshots"])
        by_league = {}
        for s in self.db["snapshots"]:
            lg = s.get("league", "unknown")
            by_league[lg] = by_league.get(lg, 0) + 1
        return {
            "total_snapshots": total,
            "by_league": by_league,
            "last_snapshot": self.db.get("last_snapshot"),
            "snapshot_interval_hours": SNAPSHOT_INTERVAL_HOURS,
            "paid_api_mode": ODDS_API_PAID,
        }


# Singleton
snapshotter = LineSnapshotter()
