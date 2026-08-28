"""
data_fetcher.py

Fetches game data from:
- ESPN public NFL endpoints
- CollegeFootballData API for NCAAF

Training goal:
- NFL: 2021 through latest completed season
- CFB: 2021 through latest completed season
- Current season may be included once meaningful completed games exist
"""

import os
import httpx
import asyncio
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

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
    NFL season year is the year the season starts.

    Example:
    Jan 2026 still belongs to the 2025 NFL season.
    """
    return CURRENT_YEAR if CURRENT_MONTH >= 8 else CURRENT_YEAR - 1


def current_cfb_season() -> int:
    """
    College football season year is the fall season year.
    """
    return CURRENT_YEAR if CURRENT_MONTH >= 8 else CURRENT_YEAR - 1


def latest_completed_nfl_season() -> int:
    """
    At the beginning of a new season, the previous season
    is the latest completed training season.
    """
    cur = current_nfl_season()

    if CURRENT_MONTH >= 8:
        return cur - 1

    return cur - 1


def latest_completed_cfb_season() -> int:
    """
    Before/early in the current fall season, use the previous
    season as the latest completed training season.
    """
    cur = current_cfb_season()

    if CURRENT_MONTH >= 8:
        return cur - 1

    return cur - 1


def nfl_training_seasons() -> list[int]:
    """
    Client requirement:
    retain training data from 2021 through latest completed season.
    """
    end = latest_completed_nfl_season()

    return list(
        range(
            TRAINING_START_YEAR,
            end + 1,
        )
    )


def cfb_training_seasons() -> list[int]:
    """
    Client requirement:
    retain training data from 2021 through latest completed season.
    """
    end = latest_completed_cfb_season()

    return list(
        range(
            TRAINING_START_YEAR,
            end + 1,
        )
    )


# Backwards-compatible helper used elsewhere in the project.
def training_seasons() -> list[int]:
    return nfl_training_seasons()


# ============================================================
# CFBD authentication
# ============================================================

def cfbd_headers() -> dict:
    if not CFBD_API_KEY:
        logger.warning(
            "CFBD_API_KEY is not configured."
        )
        return {}

    return {
        "Authorization": f"Bearer {CFBD_API_KEY}",
        "Accept": "application/json",
    }


async def test_cfbd_connection() -> dict:
    """
    Lightweight authentication test.

    Returns a small status dictionary rather than raising.
    """
    if not CFBD_API_KEY:
        return {
            "ok": False,
            "status": None,
            "error": "CFBD_API_KEY is missing",
        }

    url = f"{CFBD_BASE}/teams"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                url,
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
            "status": response.status_code,
            "records": len(data)
            if isinstance(data, list)
            else None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "error": str(exc),
        }


# ============================================================
# NFL
# ============================================================

async def get_nfl_teams() -> list[dict]:
    url = f"{ESPN_BASE}/nfl/teams"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    return [
        {
            "id": team["team"]["id"],
            "name": team["team"]["displayName"],
            "abbr": team["team"]["abbreviation"],
        }
        for team in (
            data.get(
                "sports",
                [{}],
            )[0]
            .get(
                "leagues",
                [{}],
            )[0]
            .get(
                "teams",
                [],
            )
        )
    ]


async def get_nfl_schedule(
    season: int,
    week: int,
) -> list[dict]:

    url = (
        f"{ESPN_BASE}/nfl/scoreboard"
        f"?seasontype=2"
        f"&week={week}"
        f"&limit=20"
    )

    if season < current_nfl_season():
        url += f"&dates={season}"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    games = []

    for event in data.get(
        "events",
        [],
    ):

        competition = event.get(
            "competitions",
            [{}],
        )[0]

        teams = competition.get(
            "competitors",
            [],
        )

        if len(teams) < 2:
            continue

        home = next(
            (
                team
                for team in teams
                if team.get(
                    "homeAway"
                )
                == "home"
            ),
            teams[0],
        )

        away = next(
            (
                team
                for team in teams
                if team.get(
                    "homeAway"
                )
                == "away"
            ),
            teams[1],
        )

        home_score = int(
            home.get(
                "score",
                0,
            )
            or 0
        )

        away_score = int(
            away.get(
                "score",
                0,
            )
            or 0
        )

        status = (
            event.get(
                "status",
                {},
            )
            .get(
                "type",
                {},
            )
            .get(
                "name",
                "",
            )
        )

        games.append({
            "game_id":
                event["id"],

            "home_team":
                home["team"]["displayName"],

            "home_team_id":
                home["team"]["id"],

            "away_team":
                away["team"]["displayName"],

            "away_team_id":
                away["team"]["id"],

            "date":
                event.get(
                    "date",
                    "",
                ),

            "venue":
                competition.get(
                    "venue",
                    {},
                ).get(
                    "fullName",
                    "",
                ),

            "home_score":
                home_score,

            "away_score":
                away_score,

            "status":
                status,

            "season":
                season,

            "week":
                week,
        })

    return games


async def get_nfl_historical_games(
    seasons: Optional[list[int]] = None,
) -> list[dict]:
    """
    Pull completed NFL games for model training.

    Defaults to 2021 through latest completed NFL season.
    """
    if seasons is None:
        seasons = nfl_training_seasons()

    all_games = []

    for season in seasons:

        logger.info(
            "Fetching NFL season %s...",
            season,
        )

        season_games = []

        # 18 regular-season weeks in modern NFL.
        for week in range(
            1,
            19,
        ):

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
                        or
                        game["away_score"] > 0
                    )
                    and game["status"]
                    in (
                        "STATUS_FINAL",
                        "STATUS_FINAL_OT",
                        "",
                    )
                ]

                season_games.extend(
                    completed
                )

                await asyncio.sleep(
                    0.15
                )

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
            len(
                season_games
            ),
        )

        all_games.extend(
            season_games
        )

    logger.info(
        "NFL total training games: %s",
        len(
            all_games
        ),
    )

    return all_games


async def get_nfl_schedule_upcoming(
    week: int = 1,
    season: Optional[int] = None,
) -> list[dict]:

    if season is None:
        season = current_nfl_season()

    return await get_nfl_schedule(
        season=season,
        week=week,
    )


# ============================================================
# NCAAF — CollegeFootballData
# ============================================================

async def get_cfb_teams(
    conference: Optional[str] = None,
) -> list[dict]:

    url = f"{CFBD_BASE}/teams"

    params = {}

    if conference:
        params["conference"] = conference

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            url,
            params=params,
            headers=cfbd_headers(),
        )

        response.raise_for_status()

        return response.json()


async def get_cfb_games(
    season: int,
    week: Optional[int] = None,
) -> list[dict]:

    url = f"{CFBD_BASE}/games"

    params = {
        "year": season,
        "seasonType": "regular",
    }

    if week is not None:
        params["week"] = week

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url,
            params=params,
            headers=cfbd_headers(),
        )

    if response.status_code != 200:
        logger.warning(
            "CFBD games %s returned %s: %s",
            season,
            response.status_code,
            response.text[:300],
        )

        return []

    return response.json()


async def get_cfb_sp_ratings(
    season: int,
) -> list[dict]:
    """
    Fetch SP+ ratings for a CFB season.
    """

    url = f"{CFBD_BASE}/ratings/sp"

    params = {
        "year": season,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            url,
            params=params,
            headers=cfbd_headers(),
        )

    if response.status_code != 200:
        logger.warning(
            "CFBD SP+ %s returned %s",
            season,
            response.status_code,
        )
        return []

    return response.json()


def _cfbd_game_completed(
    game: dict,
) -> bool:
    """
    Current CFBD response uses camelCase fields.

    Retain snake_case compatibility in case older cached data
    is encountered.
    """

    completed = game.get(
        "completed"
    )

    home_points = game.get(
        "homePoints",
        game.get(
            "home_points"
        ),
    )

    away_points = game.get(
        "awayPoints",
        game.get(
            "away_points"
        ),
    )

    if completed is False:
        return False

    return (
        home_points is not None
        and
        away_points is not None
    )


async def get_cfb_historical_games(
    seasons: Optional[list[int]] = None,
) -> list[dict]:
    """
    Pull completed CFB games.

    Defaults to seasons 2021 through latest completed season.
    """

    if seasons is None:
        seasons = cfb_training_seasons()

    all_games = []

    for season in seasons:

        try:
            games = await get_cfb_games(
                season=season
            )

            completed = [
                game
                for game in games
                if _cfbd_game_completed(
                    game
                )
            ]

            logger.info(
                "CFB %s: %s completed games",
                season,
                len(
                    completed
                ),
            )

            all_games.extend(
                completed
            )

            await asyncio.sleep(
                0.3
            )

        except Exception as exc:
            logger.warning(
                "CFB %s fetch failed: %s",
                season,
                exc,
            )

    logger.info(
        "CFB total training games: %s",
        len(
            all_games
        ),
    )

    return all_games


async def get_cfb_multi_season_sp(
    seasons: Optional[list[int]] = None,
) -> dict:
    """
    Build combined SP+ lookup.

    More recent seasons overwrite earlier seasons.
    """

    if seasons is None:
        seasons = cfb_training_seasons()

    from feature_engine import build_cfb_sp_lookup

    combined = {}

    for season in seasons:

        try:
            sp = await get_cfb_sp_ratings(
                season=season
            )

            season_lookup = (
                build_cfb_sp_lookup(
                    sp
                )
            )

            combined.update(
                season_lookup
            )

            await asyncio.sleep(
                0.2
            )

        except Exception as exc:
            logger.warning(
                "SP+ %s fetch failed: %s",
                season,
                exc,
            )

    logger.info(
        "CFB SP+ lookup: %s teams",
        len(
            combined
        ),
    )

    return combined


async def get_cfb_upcoming(
    season: Optional[int] = None,
    week: int = 1,
) -> list[dict]:

    if season is None:
        season = current_cfb_season()

    url = f"{CFBD_BASE}/games"

    params = {
        "year": season,
        "week": week,
        "seasonType": "regular",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            url,
            params=params,
            headers=cfbd_headers(),
        )

    if response.status_code != 200:
        logger.warning(
            "CFBD upcoming %s week %s returned %s: %s",
            season,
            week,
            response.status_code,
            response.text[:300],
        )

        return []

    return response.json()
