"""
roster_engine.py
Manages team rosters and positional group ratings using Firestore.
"""

import os
import logging
from typing import Optional
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Firebase Admin Init
# ─────────────────────────────────────────────

if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "token_uri": "https://oauth2.googleapis.com/token",
    })

    firebase_admin.initialize_app(cred)

db = firestore.client()

PLAYERS_COLLECTION = "roster_players"
TEAMS_COLLECTION = "roster_teams"
MOVES_COLLECTION = "roster_moves"

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


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


class RosterEngine:
    def reload(self):
        return None

    def _team_ref(self, team_name: str):
        return db.collection(TEAMS_COLLECTION).document(team_name)

    def _player_ref(self, player_id: str):
        return db.collection(PLAYERS_COLLECTION).document(player_id)

    def init_team(self, team_name: str, league: str, base_sp: float = 0.0):
        ref = self._team_ref(team_name)
        if not ref.get().exists:
            ref.set({
                "name": team_name,
                "league": league,
                "base_sp": base_sp,
                "groups": {g: {"rating": 50.0, "players": []} for g in POSITION_GROUPS},
                "roster_adjustment": 0.0,
                "updated_at": _now_iso(),
            })

    def set_team_base_rating(self, team_name: str, base_sp: float):
        ref = self._team_ref(team_name)
        if ref.get().exists:
            ref.update({
                "base_sp": base_sp,
                "updated_at": _now_iso(),
            })

    def _find_player_key(self, player_id_or_name: str) -> Optional[str]:
        lookup = player_id_or_name.strip().lower()
        slug = _slugify(player_id_or_name)

        if self._player_ref(player_id_or_name).get().exists:
            return player_id_or_name

        if self._player_ref(slug).get().exists:
            return slug

        docs = db.collection(PLAYERS_COLLECTION).stream()
        for doc in docs:
            player = doc.to_dict()
            if player.get("name", "").strip().lower() == lookup:
                return doc.id

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

        self.init_team(team, league)

        player = {
            "player_id": player_id,
            "name": name,
            "team": team,
            "position_group": position_group,
            "impact_score": float(impact_score),
            "league": league,
            "notes": notes,
            "updated_at": _now_iso(),
        }

        self._player_ref(player_id).set(player)

        team_doc = self._team_ref(team).get().to_dict() or {}
        groups = team_doc.get("groups", {g: {"rating": 50.0, "players": []} for g in POSITION_GROUPS})

        if position_group in groups:
            players = groups[position_group].get("players", [])
            if player_id not in players:
                players.append(player_id)
            groups[position_group]["players"] = players

        self._team_ref(team).update({
            "groups": groups,
            "updated_at": _now_iso(),
        })

        self._recalculate_team_adjustment(team)

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

        player_ref = self._player_ref(real_player_key)
        player = player_ref.get().to_dict()

        old_team = player["team"]
        old_group = player["position_group"]

        self.init_team(new_team, player["league"])

        old_team_doc = self._team_ref(old_team).get()
        if old_team_doc.exists:
            old_team_data = old_team_doc.to_dict()
            old_groups = old_team_data.get("groups", {})

            if old_group in old_groups:
                old_players = old_groups[old_group].get("players", [])
                old_groups[old_group]["players"] = [
                    pid for pid in old_players if pid != real_player_key
                ]

            self._team_ref(old_team).update({
                "groups": old_groups,
                "updated_at": _now_iso(),
            })

        new_team_doc = self._team_ref(new_team).get().to_dict() or {}
        new_groups = new_team_doc.get("groups", {g: {"rating": 50.0, "players": []} for g in POSITION_GROUPS})

        if old_group in new_groups:
            new_players = new_groups[old_group].get("players", [])
            if real_player_key not in new_players:
                new_players.append(real_player_key)
            new_groups[old_group]["players"] = new_players

        self._team_ref(new_team).update({
            "groups": new_groups,
            "updated_at": _now_iso(),
        })

        player_ref.update({
            "team": new_team,
            "notes": notes,
            "updated_at": _now_iso(),
        })

        self._recalculate_team_adjustment(old_team)
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
            "timestamp": _now_iso(),
        }

        db.collection(MOVES_COLLECTION).add(move_record)

        return {
            "move": move_record,
            "old_team_adjustment": self.get_team_adjustment(old_team),
            "new_team_adjustment": self.get_team_adjustment(new_team),
        }

    def _recalculate_team_adjustment(self, team_name: str):
        team_ref = self._team_ref(team_name)
        team_doc = team_ref.get()

        if not team_doc.exists:
            return

        team_data = team_doc.to_dict()
        groups = team_data.get("groups", {})
        weighted_delta = 0.0

        for group_name, group_info in POSITION_GROUPS.items():
            group_data = groups.get(group_name, {"players": []})
            player_ids = group_data.get("players", [])

            scores = []
            for pid in player_ids:
                player_doc = self._player_ref(pid).get()
                if player_doc.exists:
                    scores.append(float(player_doc.to_dict().get("impact_score", 50.0)))

            avg_score = sum(scores) / len(scores) if scores else 50.0
            group_data["rating"] = round(avg_score, 1)
            groups[group_name] = group_data

            weighted_delta += (avg_score - 50.0) * group_info["weight"]

        team_ref.update({
            "groups": groups,
            "roster_adjustment": round(weighted_delta * 0.1, 3),
            "updated_at": _now_iso(),
        })

    def get_team_adjustment(self, team_name: str) -> Optional[float]:
        doc = self._team_ref(team_name).get()
        if not doc.exists:
            return None
        return doc.to_dict().get("roster_adjustment", 0.0)

    def get_team_profile(self, team_name: str) -> Optional[dict]:
        doc = self._team_ref(team_name).get()
        if not doc.exists:
            return None

        team = doc.to_dict()
        groups = team.get("groups", {})

        for group_name, group_data in groups.items():
            enriched = []
            for pid in group_data.get("players", []):
                player_doc = self._player_ref(pid).get()
                if player_doc.exists:
                    p = player_doc.to_dict()
                    enriched.append({
                        "id": pid,
                        "name": p.get("name"),
                        "impact_score": p.get("impact_score"),
                        "notes": p.get("notes", ""),
                    })
            group_data["player_details"] = enriched

        team["groups"] = groups
        return team

    def get_all_teams(self) -> list[dict]:
        docs = db.collection(TEAMS_COLLECTION).stream()
        teams = []

        for doc in docs:
            data = doc.to_dict()
            data.pop("groups", None)
            teams.append(data)

        return teams

    def get_recent_moves(self, limit: int = 50) -> list[dict]:
        docs = (
            db.collection(MOVES_COLLECTION)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )

        return [{**doc.to_dict(), "id": doc.id} for doc in docs]

    def get_all_players(self, team: Optional[str] = None) -> list[dict]:
        query = db.collection(PLAYERS_COLLECTION)

        if team:
            docs = query.where("team", "==", team).stream()
        else:
            docs = query.stream()

        players = [{**doc.to_dict(), "id": doc.id} for doc in docs]

        return sorted(players, key=lambda p: p.get("impact_score", 0), reverse=True)

    def search_players(self, query: str) -> list[dict]:
        q = query.lower()
        players = self.get_all_players()

        return [
            p for p in players
            if q in p.get("name", "").lower() or q in p.get("team", "").lower()
        ]

    def get_db_stats(self) -> dict:
        teams = list(db.collection(TEAMS_COLLECTION).stream())
        players = list(db.collection(PLAYERS_COLLECTION).stream())
        moves = list(db.collection(MOVES_COLLECTION).stream())

        last_updated = None

        for doc in teams:
            updated = doc.to_dict().get("updated_at")
            if updated and (last_updated is None or updated > last_updated):
                last_updated = updated

        return {
            "teams": len(teams),
            "players": len(players),
            "moves_logged": len(moves),
            "last_updated": last_updated,
        }


roster_engine = RosterEngine()