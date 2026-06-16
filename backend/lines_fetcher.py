"""
lines_fetcher.py

Fetches Vegas betting lines (spread, total, moneyline) for NFL and CFB.

Source hierarchy:
  1. The Odds API (free tier: 500 req/month — enough for weekly pulls)
     Sign up free: https://the-odds-api.com
     Set ODDS_API_KEY env var.
  2. ESPN odds endpoint (free, less reliable, no key needed)
  3. Graceful fallback: return None for lines if both unavailable
"""

import httpx
import asyncio
import os
import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football"


# ─────────────────────────────────────────────
# The Odds API (primary — free tier)
# ─────────────────────────────────────────────

async def fetch_odds_api(sport: str) -> list[dict]:
    """
    Fetch lines from The Odds API.
    sport: 'americanfootball_nfl' or 'americanfootball_ncaaf'
    Returns list of game odds dicts.
    """
    if not ODDS_API_KEY:
        return []

    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            remaining = r.headers.get("x-requests-remaining", "?")
            logger.info(f"Odds API requests remaining: {remaining}")
            if r.status_code != 200:
                logger.warning(f"Odds API returned {r.status_code}: {r.text[:200]}")
                return []
            return r.json()
    except Exception as e:
        logger.error(f"Odds API fetch error: {e}")
        return []


def parse_odds_api_response(games: list[dict]) -> list[dict]:
    """
    Normalize Odds API response to a flat structure.
    Returns list of dicts with: home_team, away_team, spread, total, home_ml, away_ml
    """
    results = []
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        commence = g.get("commence_time", "")

        spread = None
        total = None
        home_ml = None
        away_ml = None

        # Find best bookmaker — prefer DraftKings or FanDuel
        bookmakers = g.get("bookmakers", [])
        preferred = next(
            (b for b in bookmakers if b["key"] in ("draftkings", "fanduel")),
            bookmakers[0] if bookmakers else None
        )

        if preferred:
            for market in preferred.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            spread = outcome.get("point")

                elif market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Over":
                            total = outcome.get("point")

                elif market["key"] == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            home_ml = outcome.get("price")
                        elif outcome["name"] == away:
                            away_ml = outcome.get("price")

        results.append({
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "spread": spread,       # Negative = home favored
            "total": total,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "bookmaker": preferred["title"] if preferred else None,
        })

    return results


# ─────────────────────────────────────────────
# ESPN Free Odds (fallback)
# ─────────────────────────────────────────────

async def fetch_espn_odds(league: str = "nfl", week: int = 1) -> list[dict]:
    """ESPN's unofficial odds endpoint — no key, but less reliable."""
    url = f"{ESPN_BASE}/{league}/scoreboard"
    params = {"week": week, "seasontype": 2}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return []
            data = r.json()

        results = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            odds_data = comp.get("odds", [{}])
            if not odds_data:
                continue
            odds = odds_data[0]

            teams = comp.get("competitors", [])
            home = next((t for t in teams if t["homeAway"] == "home"), {})
            away = next((t for t in teams if t["homeAway"] == "away"), {})

            results.append({
                "home_team": home.get("team", {}).get("displayName", ""),
                "away_team": away.get("team", {}).get("displayName", ""),
                "commence_time": event.get("date", ""),
                "spread": odds.get("spread"),
                "total": odds.get("overUnder"),
                "home_ml": None,
                "away_ml": None,
                "bookmaker": "ESPN",
            })

        return results
    except Exception as e:
        logger.warning(f"ESPN odds fetch error: {e}")
        return []


# ─────────────────────────────────────────────
# Unified fetcher
# ─────────────────────────────────────────────

async def get_lines(league: str = "NFL", week: int = 1) -> list[dict]:
    """
    Fetch lines using best available source.
    Returns normalized list with spread + total for each game.
    """
    sport_key = "americanfootball_nfl" if league.upper() == "NFL" else "americanfootball_ncaaf"

    if ODDS_API_KEY:
        raw = await fetch_odds_api(sport_key)
        if raw:
            return parse_odds_api_response(raw)

    # Fallback to ESPN
    logger.info("Odds API key not set — falling back to ESPN odds")
    return await fetch_espn_odds(league=league.lower(), week=week)


def build_lines_lookup(lines: list[dict]) -> dict:
    """
    Build a lookup dict keyed by (home_team, away_team) normalized strings.
    Handles team name mismatches between ESPN game data and odds source.
    """
    lookup = {}
    for line in lines:
        home = _normalize_team_name(line["home_team"])
        away = _normalize_team_name(line["away_team"])
        key = f"{home}|{away}"
        lookup[key] = line

    return lookup


def find_line(lookup: dict, home_team: str, away_team: str) -> Optional[dict]:
    """Find the line for a matchup, tolerant of name differences."""
    home_norm = _normalize_team_name(home_team)
    away_norm = _normalize_team_name(away_team)

    # Exact match
    key = f"{home_norm}|{away_norm}"
    if key in lookup:
        return lookup[key]

    # Fuzzy: check if any key contains both team names
    for k, v in lookup.items():
        parts = k.split("|")
        if len(parts) == 2:
            if _teams_match(home_norm, parts[0]) and _teams_match(away_norm, parts[1]):
                return v

    return None


def _normalize_team_name(name: str) -> str:
    """Lowercase, remove common suffixes for fuzzy matching."""
    return (
        name.lower()
        .replace(".", "")
        .replace("-", " ")
        .strip()
    )


def _teams_match(a: str, b: str) -> bool:
    """True if either name contains the other (handles 'Chiefs' vs 'Kansas City Chiefs')."""
    return a in b or b in a or _last_word(a) == _last_word(b)


def _last_word(s: str) -> str:
    parts = s.strip().split()
    return parts[-1] if parts else s
