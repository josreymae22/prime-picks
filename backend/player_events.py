"""
player_events.py

Roster/player ingestion layer for Prime Picks.

Source hierarchy:
  1. ESPN current NFL rosters (free)
  2. Manual entries via admin panel
  3. SportsData.io transactions (optional paid source)
  4. MySportsFeeds transactions (optional paid source)

ESPN roster imports use a neutral impact score of 50 by default.
This prevents newly imported players from affecting predictions until
their impact scores are intentionally adjusted.
"""

import os
import logging
from typing import Optional

import httpx

from roster_engine import roster_engine

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"

SPORTSDATA_KEY = os.getenv("SPORTSDATA_API_KEY")
MSF_KEY = os.getenv("MSF_API_KEY")
MSF_PASSWORD = os.getenv("MSF_PASSWORD")


# ============================================================
# Position normalization
# ============================================================

POSITION_TO_GROUP = {
    # Offense
    "QB": "QB",

    "RB": "RB",
    "FB": "RB",
    "HB": "RB",

    "WR": "WR",
    "FL": "WR",
    "SE": "WR",

    "TE": "TE",

    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "OL": "OL",
    "G": "OL",
    "T": "OL",

    # Defense
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",

    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",

    "CB": "CB",
    "DB": "CB",

    "S": "S",
    "SS": "S",
    "FS": "S",

    # Special teams
    "K": "K",
    "P": "K",
    "LS": "K",
}


def normalize_position(pos: str) -> str:
    """
    Convert detailed ESPN positions to Prime Picks position groups.

    Unknown positions default to LB only as a final fallback.
    """
    value = (pos or "").strip().upper()

    if value in POSITION_TO_GROUP:
        return POSITION_TO_GROUP[value]

    logger.debug("Unknown position '%s'; defaulting to LB", value)
    return "LB"


# ============================================================
# ESPN team discovery
# ============================================================

async def fetch_espn_nfl_teams() -> list[dict]:
    """
    Retrieve the current NFL team directory from ESPN.

    Returns:
        [
            {
                "id": "33",
                "name": "Baltimore Ravens",
                "abbreviation": "BAL"
            },
            ...
        ]
    """
    url = f"{ESPN_BASE}/nfl/teams"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                url,
                params={"limit": 100},
            )

        if response.status_code != 200:
            logger.warning(
                "ESPN NFL teams request returned HTTP %s",
                response.status_code,
            )
            return []

        data = response.json()

        teams = []

        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for entry in league.get("teams", []):
                    team = entry.get("team", {})

                    team_id = str(team.get("id", "")).strip()
                    name = (
                        team.get("displayName")
                        or team.get("shortDisplayName")
                        or team.get("name")
                        or ""
                    ).strip()

                    abbreviation = (
                        team.get("abbreviation") or ""
                    ).strip()

                    if not team_id or not name:
                        continue

                    teams.append({
                        "id": team_id,
                        "name": name,
                        "abbreviation": abbreviation,
                    })

        logger.info(
            "ESPN: discovered %s NFL teams",
            len(teams),
        )

        return teams

    except Exception as exc:
        logger.exception(
            "Unable to fetch ESPN NFL team list: %s",
            exc,
        )
        return []


# ============================================================
# ESPN roster fetching
# ============================================================

async def fetch_espn_nfl_roster(team_id: str) -> list[dict]:
    """
    Fetch the current ESPN roster for one NFL team.
    """
    url = f"{ESPN_BASE}/nfl/teams/{team_id}/roster"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url)

        if response.status_code != 200:
            logger.warning(
                "ESPN roster request for team %s returned HTTP %s",
                team_id,
                response.status_code,
            )
            return []

        data = response.json()

        players = []

        for athlete_group in data.get("athletes", []):
            for item in athlete_group.get("items", []):

                espn_id = str(item.get("id", "")).strip()

                if not espn_id:
                    continue

                name = (
                    item.get("fullName")
                    or item.get("displayName")
                    or ""
                ).strip()

                if not name:
                    continue

                pos = (
                    item.get("position", {})
                    .get("abbreviation", "")
                    .strip()
                )

                status_data = item.get("status", {})

                if isinstance(status_data, dict):
                    status = (
                        status_data.get("type")
                        or status_data.get("name")
                        or "active"
                    )
                else:
                    status = str(status_data or "active")

                players.append({
                    "player_id": f"espn_nfl_{espn_id}",
                    "name": name,
                    "position": pos,
                    "position_group": normalize_position(pos),
                    "jersey": item.get("jersey", ""),
                    "status": status,
                })

        logger.info(
            "ESPN: fetched %s players for NFL team %s",
            len(players),
            team_id,
        )

        return players

    except Exception as exc:
        logger.warning(
            "ESPN roster fetch failed for team %s: %s",
            team_id,
            exc,
        )
        return []


async def fetch_espn_cfb_roster(team_id: str) -> list[dict]:
    """
    Fetch the current ESPN roster for one college team.

    This function remains available for future CFB roster syncing.
    """
    url = (
        f"{ESPN_BASE}/college-football/"
        f"teams/{team_id}/roster"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url)

        if response.status_code != 200:
            return []

        data = response.json()

        players = []

        for athlete_group in data.get("athletes", []):
            for item in athlete_group.get("items", []):

                espn_id = str(item.get("id", "")).strip()

                if not espn_id:
                    continue

                name = (
                    item.get("fullName")
                    or item.get("displayName")
                    or ""
                ).strip()

                if not name:
                    continue

                pos = (
                    item.get("position", {})
                    .get("abbreviation", "")
                    .strip()
                )

                players.append({
                    "player_id": f"espn_cfb_{espn_id}",
                    "name": name,
                    "position": pos,
                    "position_group": normalize_position(pos),
                    "jersey": item.get("jersey", ""),
                    "status": "active",
                })

        return players

    except Exception as exc:
        logger.warning(
            "ESPN CFB roster fetch failed for team %s: %s",
            team_id,
            exc,
        )
        return []


# ============================================================
# ESPN full NFL roster synchronization
# ============================================================

async def sync_espn_nfl_rosters() -> dict:
    """
    Import all current NFL players from ESPN into Firestore.

    Existing ESPN players:
      - keep their current impact score
      - update roster/team when necessary

    New ESPN players:
      - receive neutral impact score 50
      - are added to Firestore

    Manual players are not deleted or overwritten.
    """

    result = {
        "source": "ESPN",
        "teams_found": 0,
        "teams_synced": 0,
        "players_found": 0,
        "players_added": 0,
        "players_updated": 0,
        "players_moved": 0,
        "errors": [],
    }

    teams = await fetch_espn_nfl_teams()

    result["teams_found"] = len(teams)

    if not teams:
        result["errors"].append(
            "ESPN returned no NFL teams."
        )
        return result

    # Get current Firestore players once instead of repeatedly.
    existing_players = roster_engine.get_all_players()

    existing_by_id = {
        str(player.get("player_id") or player.get("id")): player
        for player in existing_players
        if player.get("player_id") or player.get("id")
    }

    for team in teams:

        team_id = team["id"]
        team_name = team["name"]

        try:
            roster = await fetch_espn_nfl_roster(team_id)

            if not roster:
                logger.warning(
                    "ESPN returned no players for %s",
                    team_name,
                )
                continue

            result["teams_synced"] += 1
            result["players_found"] += len(roster)

            for player in roster:

                player_id = player["player_id"]

                try:
                    existing = existing_by_id.get(player_id)

                    # --------------------------------------------
                    # Existing ESPN player
                    # --------------------------------------------
                    if existing:

                        old_team = existing.get("team")

                        # Player changed teams.
                        if old_team and old_team != team_name:

                            roster_engine.transfer_player(
                                player_id=player_id,
                                new_team=team_name,
                                notes="ESPN automatic roster sync",
                                move_type="roster_update",
                            )

                            result["players_moved"] += 1

                        # Preserve custom impact score.
                        impact_score = float(
                            existing.get(
                                "impact_score",
                                50.0,
                            )
                        )

                        roster_engine.add_or_update_player(
                            player_id=player_id,
                            name=player["name"],
                            team=team_name,
                            position_group=player[
                                "position_group"
                            ],
                            impact_score=impact_score,
                            league="NFL",
                            notes=(
                                f"ESPN sync"
                                f" | status={player['status']}"
                                f" | jersey={player['jersey']}"
                            ),
                        )

                        result["players_updated"] += 1

                    # --------------------------------------------
                    # New ESPN player
                    # --------------------------------------------
                    else:

                        roster_engine.add_or_update_player(
                            player_id=player_id,
                            name=player["name"],
                            team=team_name,
                            position_group=player[
                                "position_group"
                            ],
                            impact_score=50.0,
                            league="NFL",
                            notes=(
                                f"ESPN sync"
                                f" | status={player['status']}"
                                f" | jersey={player['jersey']}"
                            ),
                        )

                        result["players_added"] += 1

                        existing_by_id[player_id] = {
                            "player_id": player_id,
                            "name": player["name"],
                            "team": team_name,
                            "position_group":
                                player["position_group"],
                            "impact_score": 50.0,
                            "league": "NFL",
                        }

                except Exception as exc:
                    message = (
                        f"{player.get('name', player_id)}: "
                        f"{exc}"
                    )

                    logger.exception(
                        "Error syncing player %s",
                        message,
                    )

                    result["errors"].append(message)

        except Exception as exc:
            message = f"{team_name}: {exc}"

            logger.exception(
                "Error syncing team %s",
                team_name,
            )

            result["errors"].append(message)

    return result


# ============================================================
# SportsData.io adapter
# ============================================================

async def fetch_sportsdata_nfl_transactions(
    season: Optional[str] = None,
) -> list[dict]:
    """
    Fetch NFL transactions from SportsData.io.

    Optional paid integration.
    """

    if not SPORTSDATA_KEY:
        return []

    if season is None:
        season = "2026"

    url = (
        "https://api.sportsdata.io/v3/nfl/"
        f"scores/json/Transactions/{season}"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key":
                        SPORTSDATA_KEY
                },
            )

        if response.status_code != 200:
            logger.warning(
                "SportsData returned HTTP %s",
                response.status_code,
            )
            return []

        data = response.json()

        moves = []

        for transaction in data:

            move_type = transaction.get(
                "TransactionType"
            )

            if move_type not in (
                "Signed",
                "Trade",
                "Released",
                "Waived",
            ):
                continue

            moves.append({
                "player_id":
                    f"sd_{transaction.get('PlayerID', '')}",

                "name":
                    (
                        f"{transaction.get('FirstName', '')} "
                        f"{transaction.get('LastName', '')}"
                    ).strip(),

                "from_team":
                    transaction.get(
                        "PreviousTeam",
                        "",
                    ),

                "to_team":
                    transaction.get(
                        "Team",
                        "",
                    ),

                "position":
                    transaction.get(
                        "Position",
                        "",
                    ),

                "move_type":
                    move_type,

                "date":
                    transaction.get(
                        "Date",
                        "",
                    ),
            })

        return moves

    except Exception as exc:
        logger.error(
            "SportsData transaction fetch error: %s",
            exc,
        )
        return []


async def fetch_sportsdata_cfb_transfers(
    season: Optional[str] = None,
) -> list[dict]:

    if not SPORTSDATA_KEY:
        return []

    if season is None:
        season = "2026"

    url = (
        "https://api.sportsdata.io/v3/cfb/"
        f"scores/json/Transfers/{season}"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key":
                        SPORTSDATA_KEY
                },
            )

        if response.status_code != 200:
            return []

        data = response.json()

        return [
            {
                "player_id":
                    f"sd_cfb_{item.get('PlayerID', '')}",

                "name":
                    (
                        f"{item.get('FirstName', '')} "
                        f"{item.get('LastName', '')}"
                    ).strip(),

                "from_team":
                    item.get("PreviousSchool", ""),

                "to_team":
                    item.get("School", ""),

                "position":
                    item.get("Position", ""),

                "move_type":
                    "transfer_portal",

                "date":
                    item.get("TransferDate", ""),

                "stars":
                    item.get("Stars", 0),
            }
            for item in data
        ]

    except Exception as exc:
        logger.error(
            "SportsData CFB transfer error: %s",
            exc,
        )
        return []


# ============================================================
# MySportsFeeds adapter
# ============================================================

async def fetch_msf_nfl_roster_moves(
    season: str = "2025-2026-regular",
) -> list[dict]:

    if not MSF_KEY or not MSF_PASSWORD:
        return []

    try:
        import base64

        creds = base64.b64encode(
            f"{MSF_KEY}:{MSF_PASSWORD}".encode()
        ).decode()

        url = (
            "https://api.mysportsfeeds.com/v2.1/"
            f"pull/nfl/{season}/transactions.json"
        )

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                url,
                headers={
                    "Authorization":
                        f"Basic {creds}"
                },
            )

        if response.status_code != 200:
            return []

        data = response.json()

        moves = []

        for transaction in data.get(
            "transactions",
            [],
        ):

            player = transaction.get(
                "player",
                {},
            )

            moves.append({
                "player_id":
                    f"msf_{player.get('id', '')}",

                "name":
                    player.get(
                        "fullName",
                        "",
                    ),

                "from_team":
                    transaction.get(
                        "fromTeam",
                        {},
                    ).get(
                        "abbreviation",
                        "",
                    ),

                "to_team":
                    transaction.get(
                        "toTeam",
                        {},
                    ).get(
                        "abbreviation",
                        "",
                    ),

                "position":
                    player.get(
                        "primaryPosition",
                        "",
                    ),

                "move_type":
                    transaction.get(
                        "transactionType",
                        "",
                    ),

                "date":
                    transaction.get(
                        "updatedOn",
                        "",
                    ),
            })

        return moves

    except Exception as exc:
        logger.error(
            "MySportsFeeds error: %s",
            exc,
        )
        return []


# ============================================================
# Unified ingestion
# ============================================================

async def ingest_player_moves(
    league: str = "NFL",
) -> dict:
    """
    Prime Picks roster synchronization entry point.

    NFL:
      ESPN current rosters are always synchronized first.

    Optional paid APIs may later supplement ESPN with richer
    transaction history.
    """

    league = league.upper()

    result = {
        "source": [],
        "moves_processed": 0,
        "errors": [],
    }

    # --------------------------------------------------------
    # Free ESPN synchronization
    # --------------------------------------------------------

    if league == "NFL":

        espn_result = await sync_espn_nfl_rosters()

        result["source"].append("ESPN")

        result["espn"] = espn_result

        result["moves_processed"] += (
            espn_result.get(
                "players_moved",
                0,
            )
        )

        result["errors"].extend(
            espn_result.get(
                "errors",
                [],
            )
        )

    # --------------------------------------------------------
    # SportsData optional supplement
    # --------------------------------------------------------

    if SPORTSDATA_KEY:

        result["source"].append(
            "SportsData.io"
        )

        try:
            if league == "NFL":
                paid_moves = (
                    await
                    fetch_sportsdata_nfl_transactions()
                )
            else:
                paid_moves = (
                    await
                    fetch_sportsdata_cfb_transfers()
                )

            result["sportsdata_transactions"] = len(
                paid_moves
            )

        except Exception as exc:
            result["errors"].append(
                f"SportsData error: {exc}"
            )

    # --------------------------------------------------------
    # MySportsFeeds optional supplement
    # --------------------------------------------------------

    elif MSF_KEY and MSF_PASSWORD:

        result["source"].append(
            "MySportsFeeds"
        )

        try:
            msf_moves = (
                await
                fetch_msf_nfl_roster_moves()
            )

            result["mysportsfeeds_transactions"] = (
                len(msf_moves)
            )

        except Exception as exc:
            result["errors"].append(
                f"MySportsFeeds error: {exc}"
            )

    if league != "NFL":
        result["note"] = (
            "Automatic ESPN full-roster sync is "
            "currently enabled for NFL only."
        )

    return result


# ============================================================
# Data source status
# ============================================================

def get_data_source_status() -> dict:

    if SPORTSDATA_KEY:
        active = "ESPN + SportsData.io"

    elif MSF_KEY and MSF_PASSWORD:
        active = "ESPN + MySportsFeeds"

    else:
        active = "ESPN (free) + Manual"

    return {
        "sportsdata_io":
            bool(SPORTSDATA_KEY),

        "mysportsfeeds":
            bool(
                MSF_KEY
                and MSF_PASSWORD
            ),

        "manual_entry":
            True,

        "espn_free":
            True,

        "espn_auto_sync":
            True,

        "active_source":
            active,
    }
