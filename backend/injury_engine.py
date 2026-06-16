"""
injury_engine.py

Pulls injury/status reports from ESPN for NFL and CFB.
Applies depth chart cascade: if starter is out, backup fills in.
Produces injury-adjusted team rating modifiers that feed into predictions.

Status multipliers (applied to player's impact_score):
  Active       → 1.00 (full contribution)
  Questionable → 0.80 (likely plays, but reduced snap count risk)
  Doubtful     → 0.40 (unlikely to play)
  Out          → 0.00 (does not play)
  IR / PUP     → 0.00 (season-ending)
  Suspended    → 0.00

Depth chart cascade:
  If starter (impact ≥ 70) is Out/IR, their backup (impact 30-50) fills in.
  Net effect: group rating drops by (starter_impact - backup_impact) * group_weight
  This is the "that safety just left, their secondary is cooked" model.
"""

import httpx
import asyncio
import json
import os
import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"
INJURY_DB_PATH = os.path.join(os.path.dirname(__file__), "injury_db.json")

STATUS_MULTIPLIERS = {
    "active":        1.00,
    "probable":      0.95,
    "questionable":  0.80,
    "doubtful":      0.40,
    "out":           0.00,
    "ir":            0.00,
    "pup":           0.00,
    "suspended":     0.00,
    "day-to-day":    0.85,
    "injured reserve": 0.00,
}

# Backup quality assumption when no backup is in DB
# Expressed as fraction of starter's impact score
BACKUP_QUALITY_FACTOR = 0.52


def _load_injury_db() -> dict:
    if os.path.exists(INJURY_DB_PATH):
        with open(INJURY_DB_PATH, "r") as f:
            return json.load(f)
    return {"injuries": {}, "last_updated": None}


def _save_injury_db(db: dict):
    db["last_updated"] = datetime.utcnow().isoformat()
    with open(INJURY_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


class InjuryEngine:

    def __init__(self):
        self.db = _load_injury_db()

    def reload(self):
        self.db = _load_injury_db()

    # ──────────────────────────────────────────
    # ESPN Fetchers
    # ──────────────────────────────────────────

    async def fetch_nfl_injuries(self) -> dict:
        """
        Pull current NFL injury report from ESPN.
        Returns dict keyed by team name → list of injured players.
        """
        url = f"{ESPN_BASE}/nfl/injuries"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    logger.warning(f"NFL injury endpoint returned {r.status_code}")
                    return {}
                data = r.json()

            injuries = {}
            for team_entry in data.get("injuries", []):
                team_name = team_entry.get("team", {}).get("displayName", "")
                team_injuries = []
                for inj in team_entry.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    status_raw = inj.get("status", "Active").lower()
                    status = self._normalize_status(status_raw)
                    pos = athlete.get("position", {}).get("abbreviation", "")

                    team_injuries.append({
                        "player_id": f"espn_{athlete.get('id', '')}",
                        "name": athlete.get("displayName", ""),
                        "position": pos,
                        "status": status,
                        "status_raw": status_raw,
                        "description": inj.get("longComment", inj.get("shortComment", "")),
                        "return_date": inj.get("returnDate", ""),
                        "fetched_at": datetime.utcnow().isoformat(),
                    })

                if team_injuries:
                    injuries[team_name] = team_injuries

            logger.info(f"NFL injuries fetched: {sum(len(v) for v in injuries.values())} players across {len(injuries)} teams")
            return injuries

        except Exception as e:
            logger.error(f"NFL injury fetch error: {e}")
            return {}

    async def fetch_cfb_injuries(self, team_ids: Optional[list] = None) -> dict:
        """
        Pull CFB injuries from ESPN for specific teams.
        ESPN's CFB injury endpoint requires team-by-team queries.
        """
        if not team_ids:
            return {}

        injuries = {}
        for team_id in team_ids[:30]:  # Cap at 30 to avoid rate limiting
            try:
                url = f"{ESPN_BASE}/college-football/teams/{team_id}/injuries"
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(url)
                    if r.status_code != 200:
                        continue
                    data = r.json()

                team_name = ""
                team_injuries = []
                for inj in data.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    if not team_name:
                        team_name = inj.get("team", {}).get("displayName", f"team_{team_id}")
                    status_raw = inj.get("status", "Active").lower()
                    status = self._normalize_status(status_raw)
                    pos = athlete.get("position", {}).get("abbreviation", "")

                    team_injuries.append({
                        "player_id": f"espn_cfb_{athlete.get('id', '')}",
                        "name": athlete.get("displayName", ""),
                        "position": pos,
                        "status": status,
                        "status_raw": status_raw,
                        "description": inj.get("longComment", ""),
                        "fetched_at": datetime.utcnow().isoformat(),
                    })

                if team_injuries and team_name:
                    injuries[team_name] = team_injuries

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"CFB injury fetch error for team {team_id}: {e}")

        return injuries

    def _normalize_status(self, raw: str) -> str:
        raw = raw.lower().strip()
        for key in STATUS_MULTIPLIERS:
            if key in raw:
                return key
        return "active"

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def update_injuries(self, injuries: dict, league: str):
        """Merge new injury data into persistent store."""
        for team, players in injuries.items():
            key = f"{league}:{team}"
            self.db["injuries"][key] = {
                "team": team,
                "league": league,
                "players": players,
                "updated_at": datetime.utcnow().isoformat(),
            }
        _save_injury_db(self.db)

    def get_team_injuries(self, team_name: str, league: str = "NFL") -> list:
        key = f"{league}:{team_name}"
        entry = self.db["injuries"].get(key, {})
        return entry.get("players", [])

    def get_all_injuries(self, league: Optional[str] = None) -> dict:
        result = {}
        for key, val in self.db["injuries"].items():
            if league and not key.startswith(f"{league}:"):
                continue
            result[val["team"]] = val["players"]
        return result

    # ──────────────────────────────────────────
    # Rating adjustment with depth chart cascade
    # ──────────────────────────────────────────

    def get_injury_adjustment(
        self,
        team_name: str,
        league: str,
        roster_engine,  # Pass in roster_engine instance
    ) -> dict:
        """
        Calculate injury-adjusted team rating modifier.

        Returns:
          adjustment: float (SP+ equivalent points, negative = worse)
          affected_players: list of dicts describing impact
          depth_chart_cascades: list of starter→backup substitutions
        """
        injuries = self.get_team_injuries(team_name, league)
        if not injuries:
            return {"adjustment": 0.0, "affected_players": [], "depth_chart_cascades": []}

        # Get team's full player roster from roster_engine
        team_players = roster_engine.get_all_players(team=team_name)
        player_lookup = {p["player_id"]: p for p in team_players}
        name_lookup = {p["name"].lower(): p for p in team_players}

        affected = []
        cascades = []
        total_adjustment = 0.0

        from player_events import POSITION_TO_GROUP
        from roster_engine import POSITION_GROUPS

        for inj in injuries:
            status = inj["status"]
            multiplier = STATUS_MULTIPLIERS.get(status, 1.0)
            if multiplier == 1.0:
                continue  # Active player, no adjustment needed

            # Find this player in our roster DB
            roster_player = (
                player_lookup.get(inj["player_id"]) or
                name_lookup.get(inj["name"].lower())
            )

            if not roster_player:
                # Player not in our DB — estimate impact based on position
                pos_group = POSITION_TO_GROUP.get(inj["position"].upper(), "LB")
                estimated_impact = 62.0  # Assume average starter
                roster_player = {
                    "name": inj["name"],
                    "position_group": pos_group,
                    "impact_score": estimated_impact,
                    "estimated": True,
                }

            impact = roster_player["impact_score"]
            pos_group = roster_player["position_group"]
            group_weight = POSITION_GROUPS.get(pos_group, {}).get("weight", 0.1)

            # Calculate rating loss from this injury
            effective_impact = impact * multiplier
            impact_loss = impact - effective_impact

            # Depth chart cascade: find backup in same position group
            backup = self._find_backup(team_players, pos_group, impact, inj.get("player_id", ""))
            backup_impact = backup["impact_score"] if backup else impact * BACKUP_QUALITY_FACTOR
            backup_name = backup["name"] if backup else "Depth player"

            if multiplier == 0.0:
                # Starter is out — backup fills in
                net_loss = (impact - backup_impact) * group_weight * 0.1  # Scale to SP+ pts
                cascade = {
                    "starter_out": inj["name"],
                    "starter_impact": impact,
                    "backup_in": backup_name,
                    "backup_impact": round(backup_impact, 1),
                    "position_group": pos_group,
                    "rating_impact": round(-net_loss, 3),
                }
                cascades.append(cascade)
                total_adjustment -= net_loss
            else:
                # Questionable/doubtful — partial suppression
                net_loss = impact_loss * group_weight * 0.1
                total_adjustment -= net_loss

            affected.append({
                "name": inj["name"],
                "status": status,
                "position_group": pos_group,
                "impact_score": impact,
                "multiplier": multiplier,
                "description": inj.get("description", ""),
                "estimated": roster_player.get("estimated", False),
            })

        return {
            "adjustment": round(total_adjustment, 3),
            "affected_players": affected,
            "depth_chart_cascades": cascades,
        }

    def _find_backup(self, team_players: list, pos_group: str, starter_impact: float, exclude_id: str) -> Optional[dict]:
        """
        Find the best available backup in the same position group.
        Backup = same group, lower impact than the starter, not the same player.
        """
        candidates = [
            p for p in team_players
            if p["position_group"] == pos_group
            and p["impact_score"] < starter_impact
            and p["player_id"] != exclude_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p["impact_score"])

    def get_status_summary(self, league: str = "NFL") -> dict:
        """Quick summary of all injury statuses across a league."""
        all_inj = self.get_all_injuries(league=league)
        summary = {"out": [], "doubtful": [], "questionable": []}
        for team, players in all_inj.items():
            for p in players:
                s = p.get("status", "active")
                if s in summary:
                    summary[s].append({"name": p["name"], "team": team, "position": p.get("position", "")})
        return summary


# Singleton
injury_engine = InjuryEngine()
