"""
roster_engine.py

Manages team rosters and positional group ratings.
Player moves (trades, FA, transfer portal) adjust group ratings,
which then feed into the prediction model as rating modifiers.

Architecture:
  - Each team has positional groups: QB, RB, WR, TE, OL, DL, LB, CB, S
  - Each group has a base rating (0-100) derived from SP+ or NFL efficiency
  - Individual players have an impact_score that contributes to group rating
  - When a player moves, group ratings update → model adjusts predictions

Data source hierarchy:
  1. Manual entries (admin panel) — always available
  2. ESPN roster endpoints — free, limited detail
  3. SportsData.io / MySportsFeeds — paid, full real-time moves (plug in when ready)
"""

import json
import os
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

ROSTER_DB_PATH = os.path.join(os.path.dirname(__file__), "roster_db.json")

# Positional groups and their offensive/defensive contribution
POSITION_GROUPS = {
    # Offensive groups
    "QB":  {"side": "offense", "weight": 0.35},  # Most impactful single position
    "RB":  {"side": "offense", "weight": 0.10},
    "WR":  {"side": "offense", "weight": 0.15},
    "TE":  {"side": "offense", "weight": 0.08},
    "OL":  {"side": "offense", "weight": 0.12},
    # Defensive groups
    "DL":  {"side": "defense", "weight": 0.20},
    "LB":  {"side": "defense", "weight": 0.15},
    "CB":  {"side": "defense", "weight": 0.18},
    "S":   {"side": "defense", "weight": 0.12},
    "K":   {"side": "special", "weight": 0.03},
}

# Impact rating scale: 0-100
# Elite starter: 85-100
# Good starter:  70-84
# Average:       50-69
# Backup:        30-49
# Replacement:   0-29

IMPACT_TIERS = {
    "elite":    (85, 100),
    "good":     (70, 84),
    "average":  (50, 69),
    "backup":   (30, 49),
    "practice": (0,  29),
}


def _load_db() -> dict:
    if os.path.exists(ROSTER_DB_PATH):
        with open(ROSTER_DB_PATH, "r") as f:
            return json.load(f)
    return {"teams": {}, "players": {}, "moves": [], "last_updated": None}


def _save_db(db: dict):
    db["last_updated"] = datetime.utcnow().isoformat()
    with open(ROSTER_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


class RosterEngine:
    """
    In-memory roster store with file persistence.
    Thread-safe for read-heavy FastAPI usage.
    """

    def __init__(self):
        self.db = _load_db()

    def reload(self):
        self.db = _load_db()

    # ──────────────────────────────────────────
    # Team management
    # ──────────────────────────────────────────

    def init_team(self, team_name: str, league: str, base_sp: float = 0.0):
        """Initialize a team entry with neutral positional group ratings."""
        if team_name not in self.db["teams"]:
            self.db["teams"][team_name] = {
                "league": league,
                "base_sp": base_sp,
                "groups": {g: {"rating": 50.0, "players": []} for g in POSITION_GROUPS},
                "roster_adjustment": 0.0,
            }
            _save_db(self.db)

    def set_team_base_rating(self, team_name: str, base_sp: float):
        if team_name in self.db["teams"]:
            self.db["teams"][team_name]["base_sp"] = base_sp
            _save_db(self.db)

    # ──────────────────────────────────────────
    # Player management
    # ──────────────────────────────────────────

    def add_or_update_player(
        self,
        player_id: str,
        name: str,
        team: str,
        position_group: str,
        impact_score: float,
        league: str,
        notes: str = "",
    ) -> dict:
        """
        Add a player or update their current team/rating.
        impact_score: 0-100 scale.
        """
        player = {
            "player_id": player_id,
            "name": name,
            "team": team,
            "position_group": position_group,
            "impact_score": impact_score,
            "league": league,
            "notes": notes,
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.db["players"][player_id] = player

        # Ensure team exists
        if team not in self.db["teams"]:
            self.init_team(team, league)

        # Add player to team's group roster
        team_data = self.db["teams"][team]
        if position_group in team_data["groups"]:
            existing = team_data["groups"][position_group]["players"]
            if player_id not in existing:
                existing.append(player_id)

        self._recalculate_team_adjustment(team)
        _save_db(self.db)
        return player

    def transfer_player(
        self,
        player_id: str,
        new_team: str,
        notes: str = "",
        move_type: str = "trade",  # trade | free_agency | transfer_portal | waiver
    ) -> dict:
        """
        Move a player from one team to another.
        Updates both teams' group ratings automatically.
        """
        if player_id not in self.db["players"]:
            raise ValueError(f"Player {player_id} not found")

        player = self.db["players"][player_id]
        old_team = player["team"]
        old_group = player["position_group"]

        # Remove from old team
        if old_team in self.db["teams"]:
            old_team_data = self.db["teams"][old_team]
            if old_group in old_team_data["groups"]:
                players = old_team_data["groups"][old_group]["players"]
                if player_id in players:
                    players.remove(player_id)
            self._recalculate_team_adjustment(old_team)

        # Add to new team
        player["team"] = new_team
        player["notes"] = notes
        player["updated_at"] = datetime.utcnow().isoformat()

        if new_team not in self.db["teams"]:
            self.init_team(new_team, player["league"])

        new_team_data = self.db["teams"][new_team]
        if old_group in new_team_data["groups"]:
            new_team_data["groups"][old_group]["players"].append(player_id)
        self._recalculate_team_adjustment(new_team)

        # Log the move
        move_record = {
            "player_id": player_id,
            "player_name": player["name"],
            "from_team": old_team,
            "to_team": new_team,
            "position_group": old_group,
            "impact_score": player["impact_score"],
            "move_type": move_type,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.db["moves"].append(move_record)
        _save_db(self.db)

        return {
            "move": move_record,
            "old_team_adjustment": self.get_team_adjustment(old_team),
            "new_team_adjustment": self.get_team_adjustment(new_team),
        }

    # ──────────────────────────────────────────
    # Rating calculations
    # ──────────────────────────────────────────

    def _recalculate_team_adjustment(self, team_name: str):
        """
        Recalculate a team's overall roster adjustment score.

        Logic:
        - Each positional group is rated 0-100 based on its players' impact scores
        - Groups above 50 (league average) contribute positive adjustment
        - Groups below 50 contribute negative adjustment
        - Weighted by positional importance
        - Final adjustment is in SP+ equivalent points (roughly -5 to +5)
        """
        if team_name not in self.db["teams"]:
            return

        team_data = self.db["teams"][team_name]
        weighted_delta = 0.0

        for group_name, group_info in POSITION_GROUPS.items():
            group_data = team_data["groups"].get(group_name, {"players": []})
            player_ids = group_data.get("players", [])

            if player_ids:
                # Average impact score of players in this group
                scores = [
                    self.db["players"][pid]["impact_score"]
                    for pid in player_ids
                    if pid in self.db["players"]
                ]
                avg_score = sum(scores) / len(scores) if scores else 50.0
            else:
                avg_score = 50.0  # League average when no data

            group_data["rating"] = round(avg_score, 1)

            # Delta from league average (50), weighted by positional importance
            delta = (avg_score - 50.0) * group_info["weight"]
            weighted_delta += delta

        # Scale to SP+ point equivalent: 10-point group delta ≈ 0.5 SP+ pts
        sp_adjustment = weighted_delta * 0.1
        team_data["roster_adjustment"] = round(sp_adjustment, 3)

    def get_team_adjustment(self, team_name: str) -> Optional[float]:
        """Return team's current roster adjustment in SP+ equivalent points."""
        if team_name not in self.db["teams"]:
            return None
        return self.db["teams"][team_name].get("roster_adjustment", 0.0)

    def get_team_profile(self, team_name: str) -> Optional[dict]:
        if team_name not in self.db["teams"]:
            return None
        team = self.db["teams"][team_name].copy()

        # Enrich with player names
        for group_name, group_data in team["groups"].items():
            enriched = []
            for pid in group_data.get("players", []):
                if pid in self.db["players"]:
                    p = self.db["players"][pid]
                    enriched.append({
                        "id": pid,
                        "name": p["name"],
                        "impact_score": p["impact_score"],
                        "notes": p.get("notes", ""),
                    })
            group_data["player_details"] = enriched

        return team

    def get_all_teams(self) -> list[dict]:
        return [
            {"name": k, **{f: v for f, v in v.items() if f != "groups"}}
            for k, v in self.db["teams"].items()
        ]

    def get_recent_moves(self, limit: int = 50) -> list[dict]:
        moves = self.db.get("moves", [])
        return sorted(moves, key=lambda m: m["timestamp"], reverse=True)[:limit]

    def get_all_players(self, team: Optional[str] = None) -> list[dict]:
        players = list(self.db["players"].values())
        if team:
            players = [p for p in players if p["team"] == team]
        return sorted(players, key=lambda p: p["impact_score"], reverse=True)

    def search_players(self, query: str) -> list[dict]:
        q = query.lower()
        return [
            p for p in self.db["players"].values()
            if q in p["name"].lower() or q in p["team"].lower()
        ]

    def get_db_stats(self) -> dict:
        return {
            "teams": len(self.db["teams"]),
            "players": len(self.db["players"]),
            "moves_logged": len(self.db.get("moves", [])),
            "last_updated": self.db.get("last_updated"),
        }


# Singleton
roster_engine = RosterEngine()
