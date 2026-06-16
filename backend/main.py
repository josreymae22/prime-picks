"""
main.py
FastAPI backend for Prime Picks.
Trains on 2023 + 2024 + 2025 (when available) data at startup.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from data_fetcher import (
    get_nfl_teams, get_nfl_historical_games, get_nfl_schedule_upcoming,
    get_cfb_historical_games, get_cfb_multi_season_sp, get_cfb_upcoming, get_cfb_teams,
    training_seasons, current_nfl_season, current_cfb_season,
)
from feature_engine import (
    build_nfl_team_rolling, build_nfl_matchup_features,
    build_cfb_sp_lookup, build_cfb_matchup_features, build_cfb_training_data,
    build_nfl_game_features,
)
from models import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_state = {
    "nfl_teams": [],
    "cfb_teams": [],
    "nfl_team_stats": {},
    "cfb_sp_lookup": {},
    "training_results": {},
    "training_seasons": [],
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    seasons = training_seasons()
    logger.info(f"🏈 Prime Picks starting — training on seasons: {seasons}")

    try:
        # ── NFL ──────────────────────────────────────
        logger.info("Fetching NFL teams...")
        app_state["nfl_teams"] = await get_nfl_teams()

        logger.info(f"Fetching NFL historical games ({seasons})...")
        nfl_games = await get_nfl_historical_games(seasons=seasons)
        app_state["nfl_team_stats"] = build_nfl_team_rolling(nfl_games, window=8)

        if len(nfl_games) > 20:
            training_rows = []
            for g in nfl_games:
                feats = build_nfl_matchup_features(
                    g["home_team"], g["away_team"], app_state["nfl_team_stats"]
                )
                feats["margin"] = g["home_score"] - g["away_score"]
                feats["total"] = g["home_score"] + g["away_score"]
                training_rows.append(feats)
            nfl_train_df = pd.DataFrame(training_rows)
            nfl_result = predictor.train_nfl(nfl_train_df)
            nfl_result["seasons"] = seasons
            app_state["training_results"]["nfl"] = nfl_result
            logger.info(f"NFL trained: {nfl_result}")

        # ── CFB ──────────────────────────────────────
        logger.info(f"Fetching CFB SP+ ratings ({seasons})...")
        app_state["cfb_sp_lookup"] = await get_cfb_multi_season_sp(seasons=seasons)

        logger.info(f"Fetching CFB historical games ({seasons})...")
        cfb_games = await get_cfb_historical_games(seasons=seasons)
        cfb_train_df = build_cfb_training_data(cfb_games, app_state["cfb_sp_lookup"])
        cfb_result = predictor.train_cfb(cfb_train_df)
        cfb_result["seasons"] = seasons
        app_state["training_results"]["cfb"] = cfb_result
        logger.info(f"CFB trained: {cfb_result}")

        # ── Initial injury refresh ──────────────
        logger.info("Fetching NFL injury reports...")
        try:
            from injury_engine import injury_engine
            nfl_injuries = await injury_engine.fetch_nfl_injuries()
            injury_engine.update_injuries(nfl_injuries, "NFL")
            logger.info(f"Injuries loaded: {sum(len(v) for v in nfl_injuries.values())} players")
        except Exception as e:
            logger.warning(f"Injury fetch failed (non-fatal): {e}")

        # ── Initial line snapshot ────────────────
        logger.info("Taking initial line snapshots...")
        try:
            from line_snapshotter import snapshotter
            await snapshotter.take_all_snapshots()
        except Exception as e:
            logger.warning(f"Initial snapshot failed (non-fatal): {e}")

        app_state["training_seasons"] = seasons
        app_state["ready"] = True
        logger.info(f"✅ Prime Picks ready — {len(nfl_games)} NFL + {len(cfb_games)} CFB games trained")

    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        app_state["ready"] = False

    # Start background snapshot scheduler
    snapshot_task = asyncio.create_task(snapshotter.start_scheduler())

    yield

    # Cleanup
    snapshotter.stop_scheduler()
    snapshot_task.cancel()
    try:
        await snapshot_task
    except asyncio.CancelledError:
        pass
    logger.info("Prime Picks shutting down.")


app = FastAPI(title="Prime Picks API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    league: str
    home_team: str
    away_team: str
    neutral_site: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok", "ready": app_state["ready"]}


@app.get("/status")
def status():
    return {
        "ready": app_state["ready"],
        "training_seasons": app_state["training_seasons"],
        "model_status": predictor.status(),
        "training_results": app_state["training_results"],
        "nfl_teams_loaded": len(app_state["nfl_teams"]),
        "cfb_teams_with_sp": len(app_state["cfb_sp_lookup"]),
    }


@app.get("/teams/nfl")
def nfl_teams():
    return app_state["nfl_teams"]


@app.get("/teams/cfb")
async def cfb_teams():
    if not app_state["cfb_teams"]:
        try:
            app_state["cfb_teams"] = await get_cfb_teams()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return app_state["cfb_teams"]


@app.post("/predict")
def predict(req: PredictRequest):
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Models still training. Try again in a moment.")

    league = req.league.upper()

    if league == "NFL":
        team_stats = app_state["nfl_team_stats"]
        features = build_nfl_matchup_features(
            req.home_team, req.away_team, team_stats, neutral_site=req.neutral_site
        )
        prediction = predictor.predict_nfl(features)
        key_factors = _nfl_key_factors(req.home_team, req.away_team, features, team_stats)

    elif league == "CFB":
        sp_lookup = app_state["cfb_sp_lookup"]
        features = build_cfb_matchup_features(
            req.home_team, req.away_team, sp_lookup, neutral_site=req.neutral_site
        )
        prediction = predictor.predict_cfb(features)
        key_factors = _cfb_key_factors(req.home_team, req.away_team, features, sp_lookup)

    else:
        raise HTTPException(status_code=400, detail="League must be NFL or CFB")

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "league": league,
        "prediction": prediction,
        "features": features,
        "key_factors": key_factors,
        "trained_on_seasons": app_state["training_seasons"],
    }


@app.get("/schedule/nfl")
async def nfl_schedule(week: int = 1, season: Optional[int] = None):
    try:
        games = await get_nfl_schedule_upcoming(week=week, season=season)
        return games
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/schedule/cfb")
async def cfb_schedule(week: int = 1, season: Optional[int] = None):
    try:
        games = await get_cfb_upcoming(week=week, season=season)
        return games
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


def _nfl_key_factors(home: str, away: str, features: dict, team_stats: dict) -> list[dict]:
    factors = []
    off_delta = features.get("off_delta", 0)
    if abs(off_delta) > 3:
        leader = home if off_delta > 0 else away
        factors.append({
            "label": "Offensive Edge",
            "detail": f"{leader} averages {abs(off_delta):.1f} more pts/game recently",
            "impact": "high" if abs(off_delta) > 6 else "medium",
        })

    def_delta = features.get("def_delta", 0)
    if abs(def_delta) > 3:
        leader = away if def_delta > 0 else home
        factors.append({
            "label": "Defensive Edge",
            "detail": f"{leader} allowing fewer points on average",
            "impact": "high" if abs(def_delta) > 6 else "medium",
        })

    if not features.get("neutral_site", False):
        factors.append({
            "label": "Home Field",
            "detail": f"{home} gets +2.5 pt HFA adjustment",
            "impact": "low",
        })

    margin_diff = features.get("home_margin_avg", 0) - features.get("away_margin_avg", 0)
    if abs(margin_diff) > 5:
        leader = home if margin_diff > 0 else away
        factors.append({
            "label": "Recent Form",
            "detail": f"{leader} has significantly better recent win margins",
            "impact": "high",
        })

    return factors


def _cfb_key_factors(home: str, away: str, features: dict, sp_lookup: dict) -> list[dict]:
    factors = []
    sp_diff = features.get("sp_diff", 0)

    if abs(sp_diff) > 5:
        leader = home if sp_diff > 0 else away
        factors.append({
            "label": "SP+ Rating Gap",
            "detail": f"{leader} has a {abs(sp_diff):.1f} pt SP+ advantage",
            "impact": "high" if abs(sp_diff) > 15 else "medium",
        })

    off_adv = features.get("off_def_matchup_home", 0) - features.get("off_def_matchup_away", 0)
    if abs(off_adv) > 5:
        leader = home if off_adv > 0 else away
        factors.append({
            "label": "Offensive Matchup",
            "detail": f"{leader}'s offense has a favorable matchup vs opponent defense",
            "impact": "medium",
        })

    if not features.get("neutral_site", False):
        factors.append({
            "label": "Home Field",
            "detail": f"{home} gets +3 pt HFA adjustment",
            "impact": "low",
        })

    return factors


# ─────────────────────────────────────────────
# Weekly Card Routes
# ─────────────────────────────────────────────

from card_engine import generate_weekly_card
from lines_fetcher import get_lines

@app.get("/card/{league}")
async def weekly_card(league: str, week: int = 1, season: Optional[int] = None):
    """Generate full weekly slate with predictions + line disparity rankings."""
    if not app_state["ready"]:
        raise HTTPException(status_code=503, detail="Models still training.")
    if league.upper() not in ("NFL", "CFB"):
        raise HTTPException(status_code=400, detail="League must be NFL or CFB")
    try:
        card = await generate_weekly_card(
            league=league.upper(),
            week=week,
            season=season,
            nfl_team_stats=app_state["nfl_team_stats"],
            cfb_sp_lookup=app_state["cfb_sp_lookup"],
        )
        return card
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Roster / Player Routes
# ─────────────────────────────────────────────

from roster_engine import roster_engine, POSITION_GROUPS
from player_events import ingest_player_moves, get_data_source_status
from pydantic import BaseModel as BM

class AddPlayerRequest(BM):
    player_id: str
    name: str
    team: str
    position_group: str
    impact_score: float
    league: str
    notes: str = ""

class TransferPlayerRequest(BM):
    player_id: str
    new_team: str
    move_type: str = "trade"
    notes: str = ""

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "changeme")

def verify_admin(secret: str):
    if secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/roster/status")
def roster_status():
    return {
        "db_stats": roster_engine.get_db_stats(),
        "data_sources": get_data_source_status(),
        "position_groups": list(POSITION_GROUPS.keys()),
    }

@app.get("/roster/team/{team_name}")
def team_profile(team_name: str):
    profile = roster_engine.get_team_profile(team_name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not in roster DB. Add players first.")
    return profile

@app.get("/roster/teams")
def all_teams():
    return roster_engine.get_all_teams()

@app.get("/roster/moves")
def recent_moves(limit: int = 50):
    return roster_engine.get_recent_moves(limit=limit)

@app.get("/roster/players")
def all_players(team: Optional[str] = None):
    return roster_engine.get_all_players(team=team)

@app.get("/roster/players/search")
def search_players(q: str):
    return roster_engine.search_players(q)

@app.post("/roster/player/add")
def add_player(req: AddPlayerRequest, secret: str = ""):
    verify_admin(secret)
    player = roster_engine.add_or_update_player(
        player_id=req.player_id,
        name=req.name,
        team=req.team,
        position_group=req.position_group,
        impact_score=req.impact_score,
        league=req.league,
        notes=req.notes,
    )
    return {
        "player": player,
        "team_adjustment": roster_engine.get_team_adjustment(req.team),
    }

@app.post("/roster/player/transfer")
def transfer_player(req: TransferPlayerRequest, secret: str = ""):
    verify_admin(secret)
    try:
        result = roster_engine.transfer_player(
            player_id=req.player_id,
            new_team=req.new_team,
            move_type=req.move_type,
            notes=req.notes,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/roster/sync")
async def sync_roster_moves(league: str = "NFL", secret: str = ""):
    verify_admin(secret)
    result = await ingest_player_moves(league=league)
    return result


# ─────────────────────────────────────────────
# Injury Routes
# ─────────────────────────────────────────────

from injury_engine import injury_engine

@app.get("/injuries/{league}")
async def get_injuries(league: str):
    """Get all current injury reports for a league."""
    league = league.upper()
    all_inj = injury_engine.get_all_injuries(league=league)
    summary = injury_engine.get_status_summary(league=league)
    return {
        "league": league,
        "by_team": all_inj,
        "summary": summary,
        "last_updated": injury_engine.db.get("last_updated"),
    }

@app.post("/injuries/refresh")
async def refresh_injuries(league: str = "NFL", secret: str = ""):
    verify_admin(secret)
    if league.upper() == "NFL":
        injuries = await injury_engine.fetch_nfl_injuries()
        injury_engine.update_injuries(injuries, "NFL")
    else:
        # CFB requires team IDs — use teams from app_state
        team_ids = [t["id"] for t in app_state.get("cfb_teams", [])[:30]]
        injuries = await injury_engine.fetch_cfb_injuries(team_ids)
        injury_engine.update_injuries(injuries, "CFB")
    return {"refreshed": len(injuries), "league": league}

@app.get("/injuries/team/{team_name}")
def team_injuries(team_name: str, league: str = "NFL"):
    injuries = injury_engine.get_team_injuries(team_name, league.upper())
    adj = injury_engine.get_injury_adjustment(team_name, league.upper(), roster_engine)
    return {
        "team": team_name,
        "injuries": injuries,
        "rating_adjustment": adj["adjustment"],
        "depth_chart_cascades": adj["depth_chart_cascades"],
    }


# ─────────────────────────────────────────────
# Line Movement Routes
# ─────────────────────────────────────────────

from line_snapshotter import snapshotter

@app.get("/movement/{league}")
async def line_movement(league: str, window_hours: int = 24):
    """Get line movement for all games in a league over the last N hours."""
    movements = snapshotter.get_all_movements(league.upper(), window_hours=window_hours)
    return {
        "league": league.upper(),
        "window_hours": window_hours,
        "games": movements,
        "snapshot_stats": snapshotter.get_snapshot_stats(),
    }

@app.get("/movement/game/{league}")
def game_movement(league: str, home_team: str, away_team: str, window_hours: int = 24):
    """Get detailed line movement for a specific game."""
    movement = snapshotter.get_movement(home_team, away_team, window_hours=window_hours)
    return {"home_team": home_team, "away_team": away_team, **movement}

@app.post("/movement/snapshot")
async def trigger_snapshot(league: str = "NFL", secret: str = ""):
    verify_admin(secret)
    await snapshotter.take_snapshot(league.upper())
    return {"ok": True, "stats": snapshotter.get_snapshot_stats()}
