"""
data_fetcher.py

Fetches game data from:
- ESPN public NFL endpoints
- CollegeFootballData API for NCAAF

Training policy:
- NFL: 2021 through latest completed season
- CFB: 2021 through latest completed season
- Current season is used for schedules/predictions, but is not
  included in the training dataset until it is complete.

Client requirements covered:
- Historical training data back to 2021
- Latest completed NFL season included
- Latest completed CFB season included
- Authenticated CFBD API access
"""

import os
import httpx
import asyncio
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"
CFBD_BASE = "https://api.collegefootballdata.com"

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "").strip()

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

TRAINING_START_YEAR = 2021


# ============================================================
# Season helpers
# ============================================================

def current_nfl_season() -> int:
    """
    Return the NFL season year.

    Example:
    - August 2026 -> 2026 season
    - January 2026 -> 2025 season
    """
    return CURRENT_YEAR if CURRENT_MONTH >= 8 else CURRENT_YEAR - 1


def current_cfb_season() -> int:
    """
    Return the college football season year.
    """
    return CURRENT_YEAR if CURRENT_MONTH >= 8 else CURRENT_YEAR - 1


def latest_completed_nfl_season() -> int:
    """
    Return the most recently completed NFL season.

    We intentionally do not train on an incomplete current season.
    """
    return current_nfl_season() - 1


def latest_completed_cfb_season() -> int:
    """
    Return the most recently completed CFB season.

    We intentionally do not train on an incomplete current season.
    """
    return current_cfb_season() - 1


def nfl_training_seasons() -> list[int]:
    """
    NFL model training seasons.

    Client requirement:
    Train from 2021 through the most recently completed season.
    """
    end = latest_completed_nfl_season()

    if end < TRAINING_START_YEAR:
        return []

    return list(range(TRAINING_START_YEAR, end + 1))


def cfb_training_seasons() -> list[int]:
    """
    CFB model training seasons.

    Client requirement:
    Train from 2021 through the most recently completed season.
    """
    end = latest_completed_cfb_season()

    if end < TRAINING_START_YEAR:
        return []

    return list(range(TRAINING_START_YEAR, end + 1))


def training_seasons() -> list[int]:
    """
    Backwards-compatible helper.

    Existing code that imports training_seasons() will continue
    using the NFL training range.
    """
    return nfl_training_seasons()


# ============================================================
# CFBD authentication
# ============================================================

def cfbd_headers() -> dict:
    """
    Build CollegeFootballData authorization headers.
    """
    if not CFBD_API_KEY:
        logger.warning("CFBD_API_KEY is not configured.")
        return {}

    return {
        "Authorization": f"Bearer {CFBD_API_KEY}",
        "Accept": "application/json",
    }


async def test_cfbd_connection() -> dict:
    """
    Lightweight CFBD authentication test.
    """

    if not CFBD_API_KEY:
        return {
            "ok": False,
            "status": None,
            "error": "CFBD_API_KEY is missing",
        }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{CFBD_BASE}/teams",
                headers=cfbd_headers(),
            )

        if response.status_code != 200:
            return {
                "ok": False,
                "status": response.status_code,
                "error": response.text[:500],
            }

        data = response.json()

        return {
            "ok": True,
            "status": 200,
            "records": len(data) if isinstance(data, list) else None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "error": str(exc),
        }


# ============================================================
# NFL — ESPN
# ============================================================

async def get_nfl_teams() -> list[dict]:
    """
    Fetch all NFL teams from ESPN.
    """

    url = f"{ESPN_BASE}/nfl/teams"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    leagues = (
        data.get("sports", [{}])[0]
        .get("leagues", [{}])[0]
        .get("teams", [])
    )

    return [
        {
            "id": team["team"]["id"],
            "name": team["team"]["displayName"],
            "abbr": team["team"]["abbreviation"],
        }
        for team in leagues
    ]


async def get_nfl_schedule(
    season: int,
    week: int,
) -> list[dict]:
    """
    Fetch one NFL regular-season week from ESPN.
    """

    url = (
        f"{ESPN_BASE}/nfl/scoreboard"
        f"?seasontype=2"
        f"&week={week}"
        f"&limit=50"
    )

    # Historical season selector.
    if season < current_nfl_season():
        url += f"&dates={season}"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    games = []

    for event in data.get("events", []):

        competitions = event.get("competitions", [])

        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])

        if len(competitors) < 2:
            continue

        home = next(
            (
                team
                for team in competitors
                if team.get("homeAway") == "home"
            ),
            competitors[0],
        )

        away = next(
            (
                team
                for team in competitors
                if team.get("homeAway") == "away"
            ),
            competitors[1],
        )

        try:
            home_score = int(home.get("score", 0) or 0)
        except (TypeError, ValueError):
            home_score = 0

        try:
            away_score = int(away.get("score", 0) or 0)
        except (TypeError, ValueError):
            away_score = 0

        status = (
            event.get("status", {})
            .get("type", {})
            .get("name", "")
        )

        games.append({
            "game_id": str(event.get("id", "")),
            "home_team": home.get("team", {}).get("displayName", ""),
            "home_team_id": str(home.get("team", {}).get("id", "")),
            "away_team": away.get("team", {}).get("displayName", ""),
            "away_team_id": str(away.get("team", {}).get("id", "")),
            "date": event.get("date", ""),
            "venue": competition.get("venue", {}).get("fullName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "status": status,
            "season": season,
            "week": week,
        })

    return games


async def get_nfl_historical_games(
    seasons: Optional[list[int]] = None,
) -> list[dict]:
    """
    Fetch completed NFL games for training.

    Default range:
    2021 -> latest completed NFL season.
    """

    if seasons is None:
        seasons = nfl_training_seasons()

    logger.info(
        "NFL training seasons: %s",
        seasons,
    )

    all_games = []

    for season in seasons:

        logger.info("Fetching NFL season %s...", season)

        season_games = []

        # 18 regular season weeks.
        for week in range(1, 19):

            try:
                games = await get_nfl_schedule(
                    season=season,
                    week=week,
                )

                completed = [
                    game
                    for game in games
                    if (
                        game["home_score"] > 0
                        or game["away_score"] > 0
                    )
                    and game["status"]
                    in (
                        "STATUS_FINAL",
                        "STATUS_FINAL_OT",
                        "STATUS_FINAL_2OT",
                        "",
                    )
                ]

                season_games.extend(completed)

                await asyncio.sleep(0.12)

            except Exception as exc:
                logger.warning(
                    "NFL %s week %s fetch failed: %s",
                    season,
                    week,
                    exc,
                )

        logger.info(
            "NFL %s: %s completed games",
            season,
            len(season_games),
        )

        all_games.extend(season_games)

    logger.info(
        "NFL total training games: %s",
        len(all_games),
    )

    return all_games


async def get_nfl_schedule_upcoming(
    week: int = 1,
    season: Optional[int] = None,
) -> list[dict]:
    """
    Fetch upcoming/current NFL schedule.
    """

    if season is None:
        season = current_nfl_season()

    return await get_nfl_schedule(
        season=season,
        week=week,
    )


# ============================================================
# CFB — CollegeFootballData
# ============================================================

async def get_cfb_teams(
    conference: Optional[str] = None,
) -> list[dict]:
    """
    Fetch CFB teams from CollegeFootballData.
    """

    params = {}

    if conference:
        params["conference"] = conference

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{CFBD_BASE}/teams",
            params=params,
            headers=cfbd_headers(),
        )

        response.raise_for_status()

        return response.json()


async def get_cfb_games(
    season: int,
    week: Optional[int] = None,
) -> list[dict]:
    """
    Fetch CFB games from CollegeFootballData.
    """

    params = {
        "year": season,
        "seasonType": "regular",
    }

    if week is not None:
        params["week"] = week

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{CFBD_BASE}/games",
            params=params,
            headers=cfbd_headers(),
        )

    if response.status_code != 200:

        logger.warning(
            "CFBD games %s returned %s: %s",
            season,
            response.status_code,
            response.text[:500],
        )

        return []

    return response.json()


async def get_cfb_sp_ratings(
    season: int,
) -> list[dict]:
    """
    Fetch SP+ ratings for a college football season.
    """

    params = {
        "year": season,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{CFBD_BASE}/ratings/sp",
            params=params,
            headers=cfbd_headers(),
        )

    if response.status_code != 200:

        logger.warning(
            "CFBD SP+ %s returned %s: %s",
            season,
            response.status_code,
            response.text[:300],
        )

        return []

    return response.json()


def _cfbd_game_completed(game: dict) -> bool:
    """
    Determine whether a CFBD game has a completed score.

    Supports current camelCase fields and older cached snake_case fields.
    """

    completed = game.get("completed")

    home_points = game.get(
        "homePoints",
        game.get("home_points"),
    )

    away_points = game.get(
        "awayPoints",
        game.get("away_points"),
    )

    if completed is False:
        return False

    return (
        home_points is not None
        and away_points is not None
    )


async def get_cfb_historical_games(
    seasons: Optional[list[int]] = None,
) -> list[dict]:
    """
    Fetch completed CFB games for training.

    Default range:
    2021 -> latest completed CFB season.
    """

    if seasons is None:
        seasons = cfb_training_seasons()

    logger.info(
        "CFB training seasons: %s",
        seasons,
    )

    all_games = []

    for season in seasons:

        try:
            games = await get_cfb_games(
                season=season,
            )

            completed = [
                game
                for game in games
                if _cfbd_game_completed(game)
            ]

            logger.info(
                "CFB %s: %s completed games",
                season,
                len(completed),
            )

            all_games.extend(completed)

            await asyncio.sleep(0.25)

        except Exception as exc:

            logger.warning(
                "CFB %s fetch failed: %s",
                season,
                exc,
            )

    logger.info(
        "CFB total training games: %s",
        len(all_games),
    )

    return all_games


async def get_cfb_multi_season_sp(
    seasons: Optional[list[int]] = None,
) -> dict:
    """
    Build SP+ lookup.

    Recent seasons overwrite older values for the same team.
    """

    if seasons is None:
        seasons = cfb_training_seasons()

    from feature_engine import build_cfb_sp_lookup

    combined = {}

    for season in seasons:

        try:
            ratings = await get_cfb_sp_ratings(
                season=season,
            )

            season_lookup = build_cfb_sp_lookup(
                ratings
            )

            combined.update(
                season_lookup
            )

            await asyncio.sleep(0.2)

        except Exception as exc:

            logger.warning(
                "SP+ %s fetch failed: %s",
                season,
                exc,
            )

    logger.info(
        "CFB SP+ lookup: %s teams",
        len(combined),
    )

    return combined


async def get_cfb_upcoming(
    season: Optional[int] = None,
    week: int = 1,
) -> list[dict]:
    """
    Fetch upcoming/current CFB schedule.
    """

    if season is None:
        season = current_cfb_season()

    params = {
        "year": season,
        "week": week,
        "seasonType": "regular",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{CFBD_BASE}/games",
            params=params,
            headers=cfbd_headers(),
        )

    if response.status_code != 200:

        logger.warning(
            "CFBD upcoming %s week %s returned %s: %s",
            season,
            week,
            response.status_code,
            response.text[:500],
        )

        return []

    return response.json()


# ============================================================
# Training diagnostics
# ============================================================

def get_training_config() -> dict:
    """
    Useful diagnostic helper for logs/API endpoints.
    """

    return {
        "training_start_year": TRAINING_START_YEAR,

        "nfl_current_season":
            current_nfl_season(),

        "nfl_latest_completed":
            latest_completed_nfl_season(),

        "nfl_training_seasons":
            nfl_training_seasons(),

        "cfb_current_season":
            current_cfb_season(),

        "cfb_latest_completed":
            latest_completed_cfb_season(),

        "cfb_training_seasons":
            cfb_training_seasons(),

        "cfbd_configured":
            bool(CFBD_API_KEY),
    }
