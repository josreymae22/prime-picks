"""
player_events.py

Ingestion layer for player moves and roster data.

Source hierarchy:
  1. Manual entries via admin panel (always available)
  2. ESPN roster endpoints (free, updated daily, limited move history)
  3. SportsData.io (paid — plug in SPORTSDATA_API_KEY env var)
  4. MySportsFeeds (paid — plug in MSF_API_KEY + MSF_PASSWORD env vars)

The engine normalizes all sources into a common player record format
and feeds into roster_engine.py for rating recalculation.
"""

import httpx
import asyncio
import os
import logging
from typing import Optional
from roster_engine import roster_engine, POSITION_GROUPS

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"
SPORTSDATA_KEY = os.getenv("SPORTSDATA_API_KEY")
MSF_KEY = os.getenv("MSF_API_KEY")
MSF_PASSWORD = os.getenv("MSF_PASSWORD")

# ─────────────────────────────────────────────
# Position normalization
# ─────────────────────────────────────────────

POSITION_TO_GROUP = {
    # Offense
    "QB": "QB",
    "RB": "RB", "FB": "RB", "HB": "RB",
    "WR": "WR", "FL": "WR", "SE": "WR",
    "TE": "TE",
    "OT": "OL", "OG": "OL", "C": "OL", "OL": "OL", "G": "OL", "T": "OL",
    # Defense
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "OLB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "CB", "DB": "CB",
    "S": "S", "SS": "S", "FS": "S",
    # Special
    "K": "K", "P": "K", "LS": "K",
}

def normalize_position(pos: str) -> str:
    return POSITION_TO_GROUP.get(pos.upper(), "LB")  # Default to LB if unknown


# ─────────────────────────────────────────────
# ESPN Free Roster Endpoints
# ─────────────────────────────────────────────

async def fetch_espn_nfl_roster(team_id: str) -> list[dict]:
    """Fetch current roster for an NFL team from ESPN."""
    url = f"{ESPN_BASE}/nfl/teams/{team_id}/roster"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return []
            data = r.json()

        players = []
        for athlete in data.get("athletes", []):
            for item in athlete.get("items", []):
                pos = item.get("position", {}).get("abbreviation", "")
                players.append({
                    "player_id": f"espn_nfl_{item.get('id', '')}",
                    "name": item.get("fullName", item.get("displayName", "")),
                    "position": pos,
                    "position_group": normalize_position(pos),
                    "jersey": item.get("jersey", ""),
                    "status": item.get("status", {}).get("type", "active"),
                })
        return players
    except Exception as e:
        logger.warning(f"ESPN roster fetch failed for team {team_id}: {e}")
        return []


async def fetch_espn_cfb_roster(team_id: str) -> list[dict]:
    """Fetch current roster for a CFB team from ESPN."""
    url = f"{ESPN_BASE}/college-football/teams/{team_id}/roster"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return []
            data = r.json()

        players = []
        for athlete in data.get("athletes", []):
            for item in athlete.get("items", []):
                pos = item.get("position", {}).get("abbreviation", "")
                players.append({
                    "player_id": f"espn_cfb_{item.get('id', '')}",
                    "name": item.get("fullName", item.get("displayName", "")),
                    "position": pos,
                    "position_group": normalize_position(pos),
                    "jersey": item.get("jersey", ""),
                    "status": item.get("status", {}).get("type", "active"),
                })
        return players
    except Exception as e:
        logger.warning(f"ESPN CFB roster fetch failed for team {team_id}: {e}")
        return []


# ─────────────────────────────────────────────
# SportsData.io Adapter (paid — plug in key)
# ─────────────────────────────────────────────

async def fetch_sportsdata_nfl_transactions(season: str = "2025") -> list[dict]:
    """
    Fetch NFL transactions (trades, FA signings, cuts) from SportsData.io.
    Requires SPORTSDATA_API_KEY env var.
    Sign up: https://sportsdata.io/nfl-api — starts at $9/mo.
    """
    if not SPORTSDATA_KEY:
        logger.info("SportsData.io key not configured — skipping transaction fetch")
        return []

    url = f"https://api.sportsdata.io/v3/nfl/scores/json/Transactions/{season}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers={"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY})
            if r.status_code != 200:
                logger.warning(f"SportsData transactions returned {r.status_code}")
                return []
            data = r.json()

        moves = []
        for t in data:
            if t.get("TransactionType") in ("Signed", "Trade", "Released", "Waived"):
                moves.append({
                    "player_id": f"sd_{t.get('PlayerID', '')}",
                    "name": f"{t.get('FirstName', '')} {t.get('LastName', '')}".strip(),
                    "from_team": t.get("PreviousTeam", ""),
                    "to_team": t.get("Team", ""),
                    "position": t.get("Position", ""),
                    "move_type": t.get("TransactionType", ""),
                    "date": t.get("Date", ""),
                })
        logger.info(f"SportsData: {len(moves)} NFL transactions fetched")
        return moves
    except Exception as e:
        logger.error(f"SportsData fetch error: {e}")
        return []


async def fetch_sportsdata_cfb_transfers(season: str = "2025") -> list[dict]:
    """
    Fetch CFB transfer portal entries from SportsData.io.
    Requires SPORTSDATA_API_KEY.
    """
    if not SPORTSDATA_KEY:
        return []

    url = f"https://api.sportsdata.io/v3/cfb/scores/json/Transfers/{season}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers={"Ocp-Apim-Subscription-Key": SPORTSDATA_KEY})
            if r.status_code != 200:
                return []
            data = r.json()

        moves = []
        for t in data:
            moves.append({
                "player_id": f"sd_cfb_{t.get('PlayerID', '')}",
                "name": f"{t.get('FirstName', '')} {t.get('LastName', '')}".strip(),
                "from_team": t.get("PreviousSchool", ""),
                "to_team": t.get('School', ""),
                "position": t.get("Position", ""),
                "move_type": "transfer_portal",
                "date": t.get("TransferDate", ""),
                "stars": t.get("Stars", 0),
            })
        return moves
    except Exception as e:
        logger.error(f"SportsData CFB transfer fetch error: {e}")
        return []


# ─────────────────────────────────────────────
# MySportsFeeds Adapter (paid — plug in key)
# ─────────────────────────────────────────────

async def fetch_msf_nfl_roster_moves(season: str = "2025-2026-regular") -> list[dict]:
    """
    Fetch NFL roster moves from MySportsFeeds.
    Requires MSF_API_KEY + MSF_PASSWORD env vars.
    Sign up: https://www.mysportsfeeds.com — starts at $9/mo.
    """
    if not MSF_KEY or not MSF_PASSWORD:
        return []

    url = f"https://api.mysportsfeeds.com/v2.1/pull/nfl/{season}/player_gamelogs.json"
    try:
        import base64
        creds = base64.b64encode(f"{MSF_KEY}:{MSF_PASSWORD}".encode()).decode()
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://api.mysportsfeeds.com/v2.1/pull/nfl/{season}/transactions.json",
                headers={"Authorization": f"Basic {creds}"}
            )
            if r.status_code != 200:
                return []
            data = r.json()

        moves = []
        for t in data.get("transactions", []):
            moves.append({
                "player_id": f"msf_{t.get('player', {}).get('id', '')}",
                "name": t.get("player", {}).get("fullName", ""),
                "from_team": t.get("fromTeam", {}).get("abbreviation", ""),
                "to_team": t.get("toTeam", {}).get("abbreviation", ""),
                "position": t.get("player", {}).get("primaryPosition", ""),
                "move_type": t.get("transactionType", ""),
                "date": t.get("updatedOn", ""),
            })
        return moves
    except Exception as e:
        logger.error(f"MySportsFeeds fetch error: {e}")
        return []


# ─────────────────────────────────────────────
# Unified ingestion — call this from main.py
# ─────────────────────────────────────────────

async def ingest_player_moves(league: str = "NFL") -> dict:
    """
    Pull latest player moves from all available sources
    and process them through roster_engine.
    Returns summary of what was ingested.
    """
    ingested = {"source": [], "moves_processed": 0, "errors": []}

    if SPORTSDATA_KEY:
        try:
            if league == "NFL":
                moves = await fetch_sportsdata_nfl_transactions()
            else:
                moves = await fetch_sportsdata_cfb_transfers()

            for move in moves:
                try:
                    pid = move["player_id"]
                    if pid in roster_engine.db["players"]:
                        roster_engine.transfer_player(
                            pid,
                            new_team=move["to_team"],
                            notes=move.get("date", ""),
                            move_type=move["move_type"],
                        )
                        ingested["moves_processed"] += 1
                except Exception as e:
                    ingested["errors"].append(str(e))

            ingested["source"].append("SportsData.io")
        except Exception as e:
            ingested["errors"].append(f"SportsData error: {e}")

    elif MSF_KEY:
        try:
            moves = await fetch_msf_nfl_roster_moves()
            ingested["source"].append("MySportsFeeds")
            ingested["moves_processed"] = len(moves)
        except Exception as e:
            ingested["errors"].append(f"MSF error: {e}")

    else:
        ingested["source"].append("manual_only")
        ingested["note"] = "No paid API configured. Using manual entries only. Add SPORTSDATA_API_KEY to enable auto-sync."

    return ingested


def get_data_source_status() -> dict:
    return {
        "sportsdata_io": bool(SPORTSDATA_KEY),
        "mysportsfeeds": bool(MSF_KEY and MSF_PASSWORD),
        "manual_entry": True,  # Always available
        "espn_free": True,
        "active_source": (
            "SportsData.io" if SPORTSDATA_KEY
            else "MySportsFeeds" if (MSF_KEY and MSF_PASSWORD)
            else "Manual + ESPN (free)"
        ),
    }
