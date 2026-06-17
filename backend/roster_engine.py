"""
roster_engine.py
Manages team rosters and positional group ratings.
"""

import json
import os
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

ROSTER_DB_PATH = os.path.join(os.path.dirname(__file__), "roster_db.json")

POSITION_GROUPS = {
    "QB": {"side": "offense", "weight": 0.35},
    "RB": {"side": "offense", "weight": 0.10},
    "WR": {"side": "offense", "weight": 0.15},
    "TE": {"side": "offense", "weight": 0.08},
    "OL": {"side": "offense", "weight": 0.12},
    "DL": {"side": "defense", "weight": 0.20},
    "LB": {"side": "defense", "weight": 0.15},
    "CB": {"side": "defense", "weight": 0.18},
    "S": {"side": "defense", "weight": 0.12},
    "K": {"side": "special", "weight": 0.03},
}

IMPACT_TIERS = {
    "elite": (85, 100),
    "good": (70, 84),
    "average": (50, 69),
    "backup": (30, 49),
    "practice": (0, 29),
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


def _slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


class RosterEngine:
    def __init__(self):
        self.db = _load_db()

    def reload(self):
        self.db = _load_db()

    def init_team(self, team_name: str, league: str, base_sp: float = 0.0):
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

    def _find_player_key(self, player_id_or_name: str) -> Optional[str]:
        lookup = player_id_or_name.strip().lower()

        # Exact player_id match
        if player_id_or_name in self.db["players"]:
            return player_id_or_name

        # Slug match
        slug = _slugify(player_id_or_name)
        if slug in self.db["players"]:
            return slug

        # Exact name match
        for key, player in self.db["players"].items():
            if player.get("name", "").strip().lower() == lookup:
                return key

        return None

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
        if not player_id:
            player_id = _slugify(name)

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

        if team not in self.db["teams"]:
            self.init_team(team, league)

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
        move_type: str = "trade",
    ) -> dict:
        real_player_key = self._find_player_key(player_id)

        if not real_player_key:
            raise ValueError(f"Player {player_id} not found")

        player = self.db["players"][real_player_key]
        old_team = player["team"]
        old_group = player["position_group"]

        if old_team in self.db["teams"]:
            old_team_data = self.db["teams"][old_team]
            if old_group in old_team_data["groups"]:
                players = old_team_data["groups"][old_group]["players"]
                if real_player_key in players:
                    players.remove(real_player_key)
            self._recalculate_team_adjustment(old_team)

        player["team"] = new_team
        player["notes"] = notes
        player["updated_at"] = datetime.utcnow().isoformat()

        if new_team not in self.db["teams"]:
            self.init_team(new_team, player["league"])

        new_team_data = self.db["teams"][new_team]
        if old_group in new_team_data["groups"]:
            players = new_team_data["groups"][old_group]["players"]
            if real_player_key not in players:
                players.append(real_player_key)

        self._recalculate_team_adjustment(new_team)

        move_record = {
            "player_id": real_player_key,
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

    def _recalculate_team_adjustment(self, team_name: str):
        if team_name not in self.db["teams"]:
            return

        team_data = self.db["teams"][team_name]
        weighted_delta = 0.0

        for group_name, group_info in POSITION_GROUPS.items():
            group_data = team_data["groups"].get(group_name, {"players": []})
            player_ids = group_data.get("players", [])

            if player_ids:
                scores = [
                    self.db["players"][pid]["impact_score"]
                    for pid in player_ids
                    if pid in self.db["players"]
                ]
                avg_score = sum(scores) / len(scores) if scores else 50.0
            else:
                avg_score = 50.0

            group_data["rating"] = round(avg_score, 1)
            weighted_delta += (avg_score - 50.0) * group_info["weight"]

        team_data["roster_adjustment"] = round(weighted_delta * 0.1, 3)

    def get_team_adjustment(self, team_name: str) -> Optional[float]:
        if team_name not in self.db["teams"]:
            return None
        return self.db["teams"][team_name].get("roster_adjustment", 0.0)

    def get_team_profile(self, team_name: str) -> Optional[dict]:
        if team_name not in self.db["teams"]:
            return None

        team = self.db["teams"][team_name].copy()

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


roster_engine = RosterEngine()