"""
data_fetcher.py
Fetches game data from ESPN public endpoints and CollegeFootballData.io
Trains on 2023 + 2024; pulls 2025 when available mid-season.
"""

import httpx
import asyncio
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"
CFBD_BASE = "https://api.collegefootballdata.com"

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# NFL season runs Sep–Feb; if we're past Feb, current season is same year
def current_nfl_season() -> int:
    return CURRENT_YEAR if CURRENT_MONTH >= 8 else CURRENT_YEAR - 1

def current_cfb_season() -> int:
    return CURRENT_YEAR if CURRENT_MONTH >= 8 else CURRENT_YEAR - 1

# Training seasons: last 2 completed + current if in-season
def training_seasons() -> list[int]:
    cur = current_nfl_season()
    seasons = [cur - 2, cur - 1]
    if CURRENT_MONTH >= 9:  # Current season has meaningful data after Week 3
        seasons.append(cur)
    return seasons

# ─────────────────────────────────────────────
# NFL
# ─────────────────────────────────────────────

async def get_nfl_teams() -> list[dict]:
    url = f"{ESPN_BASE}/nfl/teams"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return [
        {"id": t["team"]["id"], "name": t["team"]["displayName"], "abbr": t["team"]["abbreviation"]}
        for t in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    ]

async def get_nfl_schedule(season: int, week: int) -> list[dict]:
    url = f"{ESPN_BASE}/nfl/scoreboard?seasontype=2&week={week}&limit=20"
    # ESPN uses season year = year the season started (e.g. 2024 for 2024-25 season)
    # Add season param for historical seasons
    if season < current_nfl_season():
        url += f"&dates={season}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue
        home = next((t for t in teams if t["homeAway"] == "home"), teams[0])
        away = next((t for t in teams if t["homeAway"] == "away"), teams[1])
        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)
        status = event.get("status", {}).get("type", {}).get("name", "")
        games.append({
            "game_id": event["id"],
            "home_team": home["team"]["displayName"],
            "home_team_id": home["team"]["id"],
            "away_team": away["team"]["displayName"],
            "away_team_id": away["team"]["id"],
            "date": event.get("date", ""),
            "venue": comp.get("venue", {}).get("fullName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "status": status,
            "season": season,
        })
    return games

async def get_nfl_historical_games(seasons: Optional[list[int]] = None) -> list[dict]:
    """Pull completed NFL games across multiple seasons for model training."""
    if seasons is None:
        seasons = training_seasons()

    all_games = []
    for season in seasons:
        logger.info(f"Fetching NFL season {season}...")
        season_games = []
        for week in range(1, 19):
            try:
                games = await get_nfl_schedule(season=season, week=week)
                # Only include completed games
                completed = [
                    g for g in games
                    if (g["home_score"] > 0 or g["away_score"] > 0)
                    and g["status"] in ("STATUS_FINAL", "STATUS_FINAL_OT", "")
                ]
                season_games.extend(completed)
                await asyncio.sleep(0.15)
            except Exception as e:
                logger.warning(f"NFL {season} week {week} fetch failed: {e}")
                break  # Stop if weeks run out

        logger.info(f"NFL {season}: {len(season_games)} completed games")
        all_games.extend(season_games)

    logger.info(f"NFL total training games: {len(all_games)}")
    return all_games

async def get_nfl_schedule_upcoming(week: int = 1, season: Optional[int] = None) -> list[dict]:
    if season is None:
        season = current_nfl_season()
    return await get_nfl_schedule(season=season, week=week)

# ─────────────────────────────────────────────
# NCAAF (CollegeFootballData.io)
# ─────────────────────────────────────────────

async def get_cfb_teams(conference: Optional[str] = None) -> list[dict]:
    url = f"{CFBD_BASE}/teams"
    params = {}
    if conference:
        params["conference"] = conference
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def get_cfb_games(season: int, week: Optional[int] = None) -> list[dict]:
    url = f"{CFBD_BASE}/games"
    params = {"year": season, "seasonType": "regular"}
    if week:
        params["week"] = week
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            logger.warning(f"CFBD games {season} returned {r.status_code}")
            return []
        return r.json()

async def get_cfb_sp_ratings(season: int) -> list[dict]:
    """SP+ ratings — best single predictor for college football."""
    url = f"{CFBD_BASE}/ratings/sp"
    params = {"year": season}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            return []
        return r.json()

async def get_cfb_historical_games(seasons: Optional[list[int]] = None) -> list[dict]:
    """Pull completed CFB games across multiple seasons."""
    if seasons is None:
        seasons = training_seasons()

    all_games = []
    for season in seasons:
        try:
            games = await get_cfb_games(season=season)
            completed = [
                g for g in games
                if g.get("home_points") is not None and g.get("away_points") is not None
            ]
            logger.info(f"CFB {season}: {len(completed)} completed games")
            all_games.extend(completed)
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.warning(f"CFB {season} fetch failed: {e}")

    logger.info(f"CFB total training games: {len(all_games)}")
    return all_games

async def get_cfb_multi_season_sp(seasons: Optional[list[int]] = None) -> dict:
    """
    Build a combined SP+ lookup across multiple seasons.
    More recent seasons take precedence.
    """
    if seasons is None:
        seasons = training_seasons()

    from feature_engine import build_cfb_sp_lookup
    combined = {}
    for season in seasons:
        try:
            sp = await get_cfb_sp_ratings(season=season)
            season_lookup = build_cfb_sp_lookup(sp)
            combined.update(season_lookup)  # Later seasons overwrite earlier
            await asyncio.sleep(0.2)
        except Exception as e:
            logger.warning(f"SP+ {season} fetch failed: {e}")

    logger.info(f"CFB SP+ lookup: {len(combined)} teams")
    return combined

async def get_cfb_upcoming(season: Optional[int] = None, week: int = 1) -> list[dict]:
    if season is None:
        season = current_cfb_season()
    url = f"{CFBD_BASE}/games"
    params = {"year": season, "week": week, "seasonType": "regular"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            return []
        return r.json()
