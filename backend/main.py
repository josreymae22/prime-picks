"""
main.py

FastAPI backend for Prime Picks.

Training:
- NFL: 2021 through latest completed season
- CFB: 2021 through latest completed season
- NFL training uses chronological, leakage-safe rolling features
- CFB training uses season-specific SP+ ratings

Production:
- FastAPI binds to Render immediately
- Heavy model initialization happens in the background
- /health is available while models train
- /predict and /card return 503 until initialization completes

CFB prediction behavior:
- If both teams have SP+ data, use trained CFB model
- If either team is missing SP+ data, use a recent-performance fallback
  rather than feeding artificial zero SP+ inputs into the trained model
"""

import asyncio
import logging
import math
import os

from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_fetcher import (
    get_nfl_teams,
    get_nfl_historical_games,
    get_nfl_schedule_upcoming,
    get_cfb_historical_games,
    get_cfb_upcoming,
    get_cfb_teams,
    get_cfb_sp_ratings,
    nfl_training_seasons,
    cfb_training_seasons,
    current_cfb_season,
)

from feature_engine import (
    build_nfl_team_rolling,
    build_nfl_matchup_features,
    build_nfl_training_data,
    build_cfb_sp_lookup,
    build_cfb_matchup_features,
    build_cfb_training_data,
)

from models import predictor
from roster_engine import roster_engine, POSITION_GROUPS
from player_events import ingest_player_moves, get_data_source_status
from injury_engine import injury_engine
from line_snapshotter import snapshotter
from card_engine import generate_weekly_card


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Application state
# ============================================================

app_state = {
    "nfl_teams": [],
    "cfb_teams": [],

    "nfl_team_stats": {},
    "cfb_team_stats": {},
    "cfb_sp_lookup": {},

    "training_results": {},
    "nfl_training_seasons": [],
    "cfb_training_seasons": [],

    "ready": False,
    "initializing": False,
    "startup_error": None,

    "initialization_task": None,
    "snapshot_task": None,
}


# ============================================================
# Generic helpers
# ============================================================

def _cfb_value(
    data: dict,
    camel: str,
    snake: str,
    default=None,
):
    value = data.get(camel)

    if value is None:
        value = data.get(snake)

    return default if value is None else value


# ============================================================
# CFB historical fallback profiles
# ============================================================

def build_cfb_recent_team_stats(
    games: list[dict],
    window: int = 8,
) -> dict:
    """
    Build recent scoring profiles for CFB teams.

    Used only when the normal SP+ model cannot safely be used.

    Each profile tracks:
    - recent points scored
    - recent points allowed
    - recent margin
    - sample count

    The newest games across the historical training period
    are used, so 2025 games naturally take priority over older seasons.
    """

    team_history: dict[str, list[dict]] = {}

    sortable_games = []

    for game in games:

        home_team = _cfb_value(
            game,
            "homeTeam",
            "home_team",
            "",
        )

        away_team = _cfb_value(
            game,
            "awayTeam",
            "away_team",
            "",
        )

        home_points = _cfb_value(
            game,
            "homePoints",
            "home_points",
        )

        away_points = _cfb_value(
            game,
            "awayPoints",
            "away_points",
        )

        if (
            not home_team
            or not away_team
            or home_points is None
            or away_points is None
        ):
            continue

        try:
            home_points = float(home_points)
            away_points = float(away_points)

        except (TypeError, ValueError):
            continue

        sortable_games.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_points": home_points,
            "away_points": away_points,
            "season": int(
                _cfb_value(
                    game,
                    "season",
                    "season",
                    0,
                )
                or 0
            ),
            "week": int(
                _cfb_value(
                    game,
                    "week",
                    "week",
                    0,
                )
                or 0
            ),
            "date": (
                _cfb_value(
                    game,
                    "startDate",
                    "start_date",
                    "",
                )
                or ""
            ),
        })


    sortable_games.sort(
        key=lambda g: (
            g["season"],
            g["week"],
            g["date"],
        )
    )


    for game in sortable_games:

        home_team = game["home_team"]
        away_team = game["away_team"]

        home_points = game["home_points"]
        away_points = game["away_points"]

        team_history.setdefault(
            home_team,
            [],
        ).append({
            "pts_for": home_points,
            "pts_against": away_points,
        })

        team_history.setdefault(
            away_team,
            [],
        ).append({
            "pts_for": away_points,
            "pts_against": home_points,
        })


    profiles = {}

    for team, history in team_history.items():

        recent = history[-window:]

        if not recent:
            continue

        pts_for = [
            item["pts_for"]
            for item in recent
        ]

        pts_against = [
            item["pts_against"]
            for item in recent
        ]

        margins = [
            item["pts_for"] - item["pts_against"]
            for item in recent
        ]

        profiles[team] = {
            "avg_pts_for":
                round(
                    float(np.mean(pts_for)),
                    2,
                ),

            "avg_pts_against":
                round(
                    float(np.mean(pts_against)),
                    2,
                ),

            "avg_margin":
                round(
                    float(np.mean(margins)),
                    2,
                ),

            "games":
                len(recent),
        }


    return profiles


def _normal_cdf(
    value: float,
) -> float:
    """
    Standard normal CDF without requiring scipy here.
    """

    return (
        0.5
        *
        (
            1.0
            +
            math.erf(
                value
                /
                math.sqrt(2.0)
            )
        )
    )


def predict_cfb_fallback(
    home_team: str,
    away_team: str,
    team_stats: dict,
    neutral_site: bool = False,
) -> tuple[dict, dict]:
    """
    CFB fallback prediction for matchups lacking complete SP+ data.

    Uses recent team scoring offense/defense instead of artificial
    SP+ defaults.

    This is intentionally conservative and is not passed through
    the SP+-trained model.
    """

    default_profile = {
        "avg_pts_for": 27.0,
        "avg_pts_against": 27.0,
        "avg_margin": 0.0,
        "games": 0,
    }


    home = team_stats.get(
        home_team,
        default_profile,
    )

    away = team_stats.get(
        away_team,
        default_profile,
    )


    hfa = (
        0.0
        if neutral_site
        else 3.0
    )


    # --------------------------------------------------------
    # Base expected scoring
    #
    # Blend team's offense with opponent's defense.
    # HFA is split across the two expected scores so the final
    # margin receives the full +3 point adjustment.
    # --------------------------------------------------------

    home_score = (
        (
            home["avg_pts_for"]
            +
            away["avg_pts_against"]
        )
        / 2.0
        +
        hfa / 2.0
    )


    away_score = (
        (
            away["avg_pts_for"]
            +
            home["avg_pts_against"]
        )
        / 2.0
        -
        hfa / 2.0
    )


    # --------------------------------------------------------
    # Light recent-form adjustment
    #
    # Keep this deliberately restrained so one hot/cold stretch
    # cannot dominate the fallback.
    # --------------------------------------------------------

    form_edge = (
        home["avg_margin"]
        -
        away["avg_margin"]
    )

    form_adjustment = max(
        -6.0,
        min(
            6.0,
            form_edge * 0.20,
        ),
    )


    home_score += (
        form_adjustment / 2.0
    )

    away_score -= (
        form_adjustment / 2.0
    )


    # --------------------------------------------------------
    # Keep output in plausible football territory.
    # --------------------------------------------------------

    home_score = max(
        7.0,
        min(
            60.0,
            home_score,
        ),
    )

    away_score = max(
        7.0,
        min(
            60.0,
            away_score,
        ),
    )


    margin = (
        home_score
        -
        away_score
    )

    total = (
        home_score
        +
        away_score
    )


    margin_rmse = 14.0
    total_rmse = 18.0


    margin_lo = (
        margin
        -
        1.28
        *
        margin_rmse
    )

    margin_hi = (
        margin
        +
        1.28
        *
        margin_rmse
    )

    total_lo = (
        total
        -
        1.28
        *
        total_rmse
    )

    total_hi = (
        total
        +
        1.28
        *
        total_rmse
    )


    home_win_prob = _normal_cdf(
        margin
        /
        margin_rmse
    )


    prediction = {
        "predicted_home_score":
            round(
                home_score,
                1,
            ),

        "predicted_away_score":
            round(
                away_score,
                1,
            ),

        "predicted_margin":
            round(
                margin,
                1,
            ),

        "predicted_total":
            round(
                total,
                1,
            ),

        "margin_80_lo":
            round(
                margin_lo,
                1,
            ),

        "margin_80_hi":
            round(
                margin_hi,
                1,
            ),

        "total_80_lo":
            round(
                total_lo,
                1,
            ),

        "total_80_hi":
            round(
                total_hi,
                1,
            ),

        "home_win_prob":
            round(
                float(home_win_prob),
                3,
            ),

        # Frontend already understands this flag.
        "model_trained":
            False,

        "prediction_mode":
            "historical_fallback",
    }


    diagnostics = {
        "home_recent_profile":
            home,

        "away_recent_profile":
            away,

        "home_field_advantage":
            hfa,

        "recent_form_edge":
            round(
                form_edge,
                2,
            ),

        "recent_form_adjustment":
            round(
                form_adjustment,
                2,
            ),

        "home_history_available":
            home["games"] > 0,

        "away_history_available":
            away["games"] > 0,
    }


    return (
        prediction,
        diagnostics,
    )


# ============================================================
# Startup helpers
# ============================================================

async def build_current_cfb_sp_lookup(
    training_seasons: list[int],
) -> dict:
    """
    Build SP+ lookup used for current/live predictions.

    Try current season first.
    Fall back to latest completed season if current ratings
    are not yet available.
    """

    current_season = (
        current_cfb_season()
    )

    logger.info(
        "Fetching current CFB SP+ ratings for %s...",
        current_season,
    )

    try:

        ratings = (
            await get_cfb_sp_ratings(
                current_season
            )
        )

        lookup = (
            build_cfb_sp_lookup(
                ratings
            )
        )

        # build_cfb_sp_lookup contains the internal
        # __normalized__ helper key, so check actual team records.
        actual_team_count = len([
            key
            for key in lookup.keys()
            if not str(key).startswith("__")
        ])

        if actual_team_count > 0:

            logger.info(
                "Current CFB SP+ loaded: %s teams for %s",
                actual_team_count,
                current_season,
            )

            return lookup


    except Exception as exc:

        logger.warning(
            "Current CFB SP+ %s unavailable: %s",
            current_season,
            exc,
        )


    if training_seasons:

        fallback_season = (
            training_seasons[-1]
        )

        logger.info(
            "Falling back to CFB SP+ season %s...",
            fallback_season,
        )

        try:

            ratings = (
                await get_cfb_sp_ratings(
                    fallback_season
                )
            )

            lookup = (
                build_cfb_sp_lookup(
                    ratings
                )
            )

            actual_team_count = len([
                key
                for key in lookup.keys()
                if not str(key).startswith("__")
            ])

            logger.info(
                "Fallback CFB SP+ loaded: %s teams for %s",
                actual_team_count,
                fallback_season,
            )

            return lookup


        except Exception as exc:

            logger.warning(
                "Fallback CFB SP+ %s failed: %s",
                fallback_season,
                exc,
            )


    return {}


# ============================================================
# Background initialization
# ============================================================

async def initialize_prime_picks():
    """
    Heavy startup work.

    Runs as a background asyncio task AFTER FastAPI has
    been allowed to bind to Render's HTTP port.
    """

    app_state["initializing"] = True
    app_state["ready"] = False
    app_state["startup_error"] = None


    nfl_seasons = (
        nfl_training_seasons()
    )

    cfb_seasons = (
        cfb_training_seasons()
    )


    app_state[
        "nfl_training_seasons"
    ] = nfl_seasons

    app_state[
        "cfb_training_seasons"
    ] = cfb_seasons


    logger.info(
        "🏈 Prime Picks background initialization starting"
    )

    logger.info(
        "NFL training seasons: %s",
        nfl_seasons,
    )

    logger.info(
        "CFB training seasons: %s",
        cfb_seasons,
    )


    try:

        # ====================================================
        # NFL TEAMS
        # ====================================================

        logger.info(
            "Fetching NFL teams..."
        )

        app_state["nfl_teams"] = (
            await get_nfl_teams()
        )

        logger.info(
            "NFL teams loaded: %s",
            len(
                app_state[
                    "nfl_teams"
                ]
            ),
        )


        # ====================================================
        # NFL HISTORICAL DATA
        # ====================================================

        logger.info(
            "Fetching NFL historical games (%s)...",
            nfl_seasons,
        )

        nfl_games = (
            await get_nfl_historical_games(
                seasons=nfl_seasons
            )
        )

        logger.info(
            "NFL historical games loaded: %s",
            len(nfl_games),
        )


        # ====================================================
        # NFL CURRENT TEAM FORM
        # ====================================================

        app_state[
            "nfl_team_stats"
        ] = build_nfl_team_rolling(
            nfl_games,
            window=8,
        )

        logger.info(
            "NFL current team profiles: %s",
            len(
                app_state[
                    "nfl_team_stats"
                ]
            ),
        )


        # ====================================================
        # NFL TRAINING
        # ====================================================

        if len(nfl_games) > 20:

            logger.info(
                "Building chronological NFL training features..."
            )

            nfl_train_df = (
                build_nfl_training_data(
                    nfl_games,
                    window=8,
                )
            )

            logger.info(
                "NFL training rows generated: %s",
                len(nfl_train_df),
            )

            nfl_result = (
                predictor.train_nfl(
                    nfl_train_df
                )
            )

            nfl_result[
                "seasons"
            ] = nfl_seasons

            app_state[
                "training_results"
            ][
                "nfl"
            ] = nfl_result

            logger.info(
                "NFL trained: %s",
                nfl_result,
            )

        else:

            logger.warning(
                "NFL historical dataset too small to train."
            )


        # ====================================================
        # CFB HISTORICAL DATA
        # ====================================================

        logger.info(
            "Fetching CFB historical games (%s)...",
            cfb_seasons,
        )

        cfb_games = (
            await get_cfb_historical_games(
                seasons=cfb_seasons
            )
        )

        logger.info(
            "CFB historical games loaded: %s",
            len(cfb_games),
        )


        # ====================================================
        # CFB FALLBACK TEAM PROFILES
        # ====================================================

        logger.info(
            "Building CFB recent team scoring profiles..."
        )

        app_state[
            "cfb_team_stats"
        ] = build_cfb_recent_team_stats(
            cfb_games,
            window=8,
        )

        logger.info(
            "CFB recent team profiles loaded: %s",
            len(
                app_state[
                    "cfb_team_stats"
                ]
            ),
        )


        # ====================================================
        # CFB SEASON-SPECIFIC TRAINING
        # ====================================================

        cfb_training_frames = []
        cfb_training_diagnostics = {}


        for season in cfb_seasons:

            logger.info(
                "Building CFB training data for %s...",
                season,
            )

            season_games = [
                game
                for game in cfb_games
                if game.get(
                    "season"
                ) == season
            ]

            logger.info(
                "CFB %s raw games for training: %s",
                season,
                len(
                    season_games
                ),
            )


            try:

                season_sp_ratings = (
                    await get_cfb_sp_ratings(
                        season
                    )
                )

                season_sp_lookup = (
                    build_cfb_sp_lookup(
                        season_sp_ratings
                    )
                )

                actual_sp_teams = len([
                    key
                    for key
                    in season_sp_lookup.keys()
                    if not str(
                        key
                    ).startswith(
                        "__"
                    )
                ])

                logger.info(
                    "CFB %s SP+ teams: %s",
                    season,
                    actual_sp_teams,
                )


                season_df = (
                    build_cfb_training_data(
                        season_games,
                        season_sp_lookup,
                    )
                )


                cfb_training_diagnostics[
                    season
                ] = {
                    "games":
                        len(
                            season_games
                        ),

                    "training_rows":
                        len(
                            season_df
                        ),

                    "sp_teams":
                        actual_sp_teams,

                    "skipped_missing_score":
                        season_df.attrs.get(
                            "skipped_missing_score",
                            0,
                        ),

                    "skipped_missing_team":
                        season_df.attrs.get(
                            "skipped_missing_team",
                            0,
                        ),

                    "skipped_missing_sp":
                        season_df.attrs.get(
                            "skipped_missing_sp",
                            0,
                        ),

                    "skipped_home_sp":
                        season_df.attrs.get(
                            "skipped_home_sp",
                            0,
                        ),

                    "skipped_away_sp":
                        season_df.attrs.get(
                            "skipped_away_sp",
                            0,
                        ),
                }


                logger.info(
                    (
                        "CFB %s training rows: %s "
                        "(missing SP+ skipped: %s)"
                    ),
                    season,
                    len(
                        season_df
                    ),
                    season_df.attrs.get(
                        "skipped_missing_sp",
                        0,
                    ),
                )


                if not season_df.empty:

                    cfb_training_frames.append(
                        season_df
                    )


            except Exception as exc:

                logger.warning(
                    "CFB %s training build failed: %s",
                    season,
                    exc,
                )


        # ====================================================
        # COMBINE CFB TRAINING DATA
        # ====================================================

        if cfb_training_frames:

            cfb_train_df = (
                pd.concat(
                    cfb_training_frames,
                    ignore_index=True,
                )
            )

        else:

            cfb_train_df = (
                pd.DataFrame()
            )


        logger.info(
            "CFB total model training rows: %s",
            len(
                cfb_train_df
            ),
        )


        # ====================================================
        # TRAIN CFB MODEL
        # ====================================================

        cfb_result = (
            predictor.train_cfb(
                cfb_train_df
            )
        )

        cfb_result[
            "seasons"
        ] = cfb_seasons

        cfb_result[
            "season_diagnostics"
        ] = cfb_training_diagnostics


        app_state[
            "training_results"
        ][
            "cfb"
        ] = cfb_result


        logger.info(
            "CFB trained: %s",
            cfb_result,
        )


        # ====================================================
        # CURRENT CFB SP+
        # ====================================================

        app_state[
            "cfb_sp_lookup"
        ] = (
            await build_current_cfb_sp_lookup(
                cfb_seasons
            )
        )


        # ====================================================
        # NFL INJURIES
        # ====================================================

        logger.info(
            "Fetching NFL injury reports..."
        )

        try:

            nfl_injuries = (
                await injury_engine.fetch_nfl_injuries()
            )

            injury_engine.update_injuries(
                nfl_injuries,
                "NFL",
            )

            logger.info(
                "Injuries loaded: %s players",
                sum(
                    len(players)
                    for players
                    in nfl_injuries.values()
                ),
            )

        except Exception as exc:

            logger.warning(
                "Injury fetch failed (non-fatal): %s",
                exc,
            )


        # ====================================================
        # INITIAL LINE SNAPSHOT
        # ====================================================

        logger.info(
            "Taking initial line snapshots..."
        )

        try:

            await snapshotter.take_all_snapshots()

        except Exception as exc:

            logger.warning(
                "Initial snapshot failed (non-fatal): %s",
                exc,
            )


        # ====================================================
        # START SNAPSHOT SCHEDULER
        # ====================================================

        try:

            if (
                app_state[
                    "snapshot_task"
                ]
                is None
                or
                app_state[
                    "snapshot_task"
                ].done()
            ):

                app_state[
                    "snapshot_task"
                ] = (
                    asyncio.create_task(
                        snapshotter.start_scheduler()
                    )
                )

                logger.info(
                    "Line snapshot scheduler started."
                )

        except Exception as exc:

            logger.warning(
                "Snapshot scheduler failed to start: %s",
                exc,
            )


        # ====================================================
        # READY
        # ====================================================

        app_state[
            "ready"
        ] = True

        app_state[
            "startup_error"
        ] = None


        logger.info(
            "✅ Prime Picks ready"
        )

        logger.info(
            "NFL games available for training: %s",
            len(
                nfl_games
            ),
        )

        logger.info(
            "CFB games available for training: %s",
            len(
                cfb_games
            ),
        )

        logger.info(
            "NFL actual training rows: %s",
            app_state[
                "training_results"
            ].get(
                "nfl",
                {},
            ).get(
                "n_samples",
                0,
            ),
        )

        logger.info(
            "CFB actual training rows: %s",
            app_state[
                "training_results"
            ].get(
                "cfb",
                {},
            ).get(
                "n_samples",
                0,
            ),
        )

        logger.info(
            "CFB fallback team profiles: %s",
            len(
                app_state[
                    "cfb_team_stats"
                ]
            ),
        )


    except asyncio.CancelledError:

        logger.info(
            "Prime Picks initialization task cancelled."
        )

        raise


    except Exception as exc:

        logger.error(
            "Background initialization error: %s",
            exc,
            exc_info=True,
        )

        app_state[
            "ready"
        ] = False

        app_state[
            "startup_error"
        ] = str(
            exc
        )


    finally:

        app_state[
            "initializing"
        ] = False


# ============================================================
# Application lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "🚀 Prime Picks web service starting"
    )

    app_state[
        "ready"
    ] = False

    app_state[
        "initializing"
    ] = True


    initialization_task = (
        asyncio.create_task(
            initialize_prime_picks()
        )
    )

    app_state[
        "initialization_task"
    ] = initialization_task


    # Yield immediately so Render can detect the HTTP port.
    yield


    logger.info(
        "Prime Picks shutting down."
    )


    try:

        snapshotter.stop_scheduler()

    except Exception as exc:

        logger.warning(
            "Snapshot scheduler stop error: %s",
            exc,
        )


    snapshot_task = (
        app_state.get(
            "snapshot_task"
        )
    )


    if snapshot_task:

        snapshot_task.cancel()

        try:

            await snapshot_task

        except asyncio.CancelledError:

            pass

        except Exception as exc:

            logger.warning(
                "Snapshot scheduler shutdown error: %s",
                exc,
            )


    if (
        initialization_task
        and
        not initialization_task.done()
    ):

        initialization_task.cancel()

        try:

            await initialization_task

        except asyncio.CancelledError:

            pass

        except Exception as exc:

            logger.warning(
                "Initialization shutdown error: %s",
                exc,
            )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Prime Picks API",
    version="1.3.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request models
# ============================================================

class PredictRequest(BaseModel):
    league: str
    home_team: str
    away_team: str
    neutral_site: Optional[bool] = False


class AddPlayerRequest(BaseModel):
    player_id: str
    name: str
    team: str
    position_group: str
    impact_score: float
    league: str
    notes: str = ""


class TransferPlayerRequest(BaseModel):
    player_id: str
    new_team: str
    move_type: str = "trade"
    notes: str = ""


# ============================================================
# Security
# ============================================================

ADMIN_SECRET = os.getenv(
    "ADMIN_SECRET",
    "changeme",
)


def verify_admin(
    secret: str,
):

    if secret != ADMIN_SECRET:

        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


# ============================================================
# Health / status
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok",

        "ready":
            app_state[
                "ready"
            ],

        "initializing":
            app_state[
                "initializing"
            ],

        "startup_error":
            app_state[
                "startup_error"
            ],
    }


@app.get("/status")
def status():

    cfb_sp_count = len([
        key
        for key
        in app_state[
            "cfb_sp_lookup"
        ].keys()
        if not str(
            key
        ).startswith(
            "__"
        )
    ])

    return {
        "ready":
            app_state[
                "ready"
            ],

        "initializing":
            app_state[
                "initializing"
            ],

        "startup_error":
            app_state[
                "startup_error"
            ],

        "nfl_training_seasons":
            app_state[
                "nfl_training_seasons"
            ],

        "cfb_training_seasons":
            app_state[
                "cfb_training_seasons"
            ],

        "model_status":
            predictor.status(),

        "training_results":
            app_state[
                "training_results"
            ],

        "nfl_teams_loaded":
            len(
                app_state[
                    "nfl_teams"
                ]
            ),

        "cfb_teams_with_sp":
            cfb_sp_count,

        "cfb_fallback_team_profiles":
            len(
                app_state[
                    "cfb_team_stats"
                ]
            ),
    }


# ============================================================
# Teams
# ============================================================

@app.get("/teams/nfl")
def nfl_teams():

    return app_state[
        "nfl_teams"
    ]


@app.get("/teams/cfb")
async def cfb_teams():

    if not app_state[
        "cfb_teams"
    ]:

        try:

            app_state[
                "cfb_teams"
            ] = (
                await get_cfb_teams()
            )

        except Exception as exc:

            raise HTTPException(
                status_code=503,
                detail=str(
                    exc
                ),
            )


    return app_state[
        "cfb_teams"
    ]


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(
    req: PredictRequest,
):

    if not app_state[
        "ready"
    ]:

        raise HTTPException(
            status_code=503,
            detail=(
                "Models are still initializing. "
                "Check /health and try again shortly."
            ),
        )


    league = (
        req.league.upper()
    )


    if league == "NFL":

        team_stats = (
            app_state[
                "nfl_team_stats"
            ]
        )

        features = (
            build_nfl_matchup_features(
                req.home_team,
                req.away_team,
                team_stats,
                neutral_site=
                    bool(
                        req.neutral_site
                    ),
            )
        )

        prediction = (
            predictor.predict_nfl(
                features
            )
        )

        key_factors = (
            _nfl_key_factors(
                req.home_team,
                req.away_team,
                features,
                team_stats,
            )
        )

        prediction_mode = (
            "trained_model"
            if prediction.get(
                "model_trained"
            )
            else "baseline"
        )


    elif league == "CFB":

        sp_lookup = (
            app_state[
                "cfb_sp_lookup"
            ]
        )

        features = (
            build_cfb_matchup_features(
                req.home_team,
                req.away_team,
                sp_lookup,
                neutral_site=
                    bool(
                        req.neutral_site
                    ),
            )
        )


        # ====================================================
        # Complete SP+ coverage -> trained model
        # ====================================================

        if features.get(
            "sp_data_complete",
            False,
        ):

            prediction = (
                predictor.predict_cfb(
                    features
                )
            )

            prediction_mode = (
                "trained_sp_model"
            )

            fallback_diagnostics = None


        # ====================================================
        # Missing SP+ -> historical fallback
        # ====================================================

        else:

            (
                prediction,
                fallback_diagnostics,
            ) = predict_cfb_fallback(
                req.home_team,
                req.away_team,
                app_state[
                    "cfb_team_stats"
                ],
                neutral_site=
                    bool(
                        req.neutral_site
                    ),
            )

            prediction_mode = (
                "historical_fallback"
            )


        key_factors = (
            _cfb_key_factors(
                req.home_team,
                req.away_team,
                features,
                sp_lookup,
                fallback_diagnostics=
                    fallback_diagnostics,
            )
        )


    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "League must be NFL or CFB"
            ),
        )


    response = {
        "home_team":
            req.home_team,

        "away_team":
            req.away_team,

        "league":
            league,

        "prediction_mode":
            prediction_mode,

        "prediction":
            prediction,

        "features":
            features,

        "key_factors":
            key_factors,

        "trained_on_seasons":
            (
                app_state[
                    "nfl_training_seasons"
                ]
                if league == "NFL"
                else
                app_state[
                    "cfb_training_seasons"
                ]
            ),
    }


    if (
        league == "CFB"
        and
        prediction_mode
        ==
        "historical_fallback"
    ):

        response[
            "fallback_diagnostics"
        ] = (
            fallback_diagnostics
        )


    return response


# ============================================================
# Schedules
# ============================================================

@app.get("/schedule/nfl")
async def nfl_schedule(
    week: int = 1,
    season: Optional[int] = None,
):

    try:

        return (
            await get_nfl_schedule_upcoming(
                week=week,
                season=season,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(
                exc
            ),
        )


@app.get("/schedule/cfb")
async def cfb_schedule(
    week: int = 1,
    season: Optional[int] = None,
):

    try:

        return (
            await get_cfb_upcoming(
                week=week,
                season=season,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(
                exc
            ),
        )


# ============================================================
# Key factors
# ============================================================

def _nfl_key_factors(
    home: str,
    away: str,
    features: dict,
    team_stats: dict,
) -> list[dict]:

    factors = []


    off_delta = features.get(
        "off_delta",
        0,
    )

    if abs(
        off_delta
    ) > 3:

        leader = (
            home
            if off_delta > 0
            else away
        )

        factors.append({
            "label":
                "Offensive Edge",

            "detail":
                (
                    f"{leader} averages "
                    f"{abs(off_delta):.1f} "
                    f"more pts/game recently"
                ),

            "impact":
                (
                    "high"
                    if abs(
                        off_delta
                    ) > 6
                    else "medium"
                ),
        })


    def_delta = features.get(
        "def_delta",
        0,
    )

    if abs(
        def_delta
    ) > 3:

        leader = (
            away
            if def_delta > 0
            else home
        )

        factors.append({
            "label":
                "Defensive Edge",

            "detail":
                (
                    f"{leader} allowing "
                    f"fewer points on average"
                ),

            "impact":
                (
                    "high"
                    if abs(
                        def_delta
                    ) > 6
                    else "medium"
                ),
        })


    if not features.get(
        "neutral_site",
        False,
    ):

        factors.append({
            "label":
                "Home Field",

            "detail":
                (
                    f"{home} gets "
                    f"+2.5 pt HFA adjustment"
                ),

            "impact":
                "low",
        })


    margin_diff = (
        features.get(
            "home_margin_avg",
            0,
        )
        -
        features.get(
            "away_margin_avg",
            0,
        )
    )


    if abs(
        margin_diff
    ) > 5:

        leader = (
            home
            if margin_diff > 0
            else away
        )

        factors.append({
            "label":
                "Recent Form",

            "detail":
                (
                    f"{leader} has "
                    f"significantly better "
                    f"recent win margins"
                ),

            "impact":
                "high",
        })


    return factors


def _cfb_key_factors(
    home: str,
    away: str,
    features: dict,
    sp_lookup: dict,
    fallback_diagnostics: Optional[dict] = None,
) -> list[dict]:

    factors = []


    sp_data_complete = (
        features.get(
            "sp_data_complete",
            False,
        )
    )

    home_has_sp = (
        features.get(
            "home_has_sp_data",
            False,
        )
    )

    away_has_sp = (
        features.get(
            "away_has_sp_data",
            False,
        )
    )


    # ========================================================
    # Missing SP+ warning
    # ========================================================

    if not sp_data_complete:

        missing = []

        if not home_has_sp:
            missing.append(
                home
            )

        if not away_has_sp:
            missing.append(
                away
            )


        if missing:

            factors.append({
                "label":
                    "SP+ Data Limited",

                "detail":
                    (
                        "No current SP+ rating is available for "
                        f"{', '.join(missing)}. "
                        "Prime Picks used recent scoring performance "
                        "instead of manufacturing an SP+ rating."
                    ),

                "impact":
                    "low",
            })


    # ========================================================
    # Normal SP+ factors
    # ========================================================

    if sp_data_complete:

        sp_diff = (
            features.get(
                "sp_diff",
                0,
            )
        )


        if abs(
            sp_diff
        ) > 5:

            leader = (
                home
                if sp_diff > 0
                else away
            )

            factors.append({
                "label":
                    "SP+ Rating Gap",

                "detail":
                    (
                        f"{leader} has a "
                        f"{abs(sp_diff):.1f} "
                        "pt SP+ advantage"
                    ),

                "impact":
                    (
                        "high"
                        if abs(
                            sp_diff
                        ) > 15
                        else "medium"
                    ),
            })


        off_adv = (
            features.get(
                "off_def_matchup_home",
                0,
            )
            -
            features.get(
                "off_def_matchup_away",
                0,
            )
        )


        if abs(
            off_adv
        ) > 5:

            leader = (
                home
                if off_adv > 0
                else away
            )

            factors.append({
                "label":
                    "Offensive Matchup",

                "detail":
                    (
                        f"{leader}'s offense has "
                        "a favorable matchup vs "
                        "opponent defense"
                    ),

                "impact":
                    "medium",
            })


    # ========================================================
    # Fallback recent-form factors
    # ========================================================

    elif fallback_diagnostics:

        home_profile = (
            fallback_diagnostics.get(
                "home_recent_profile",
                {},
            )
        )

        away_profile = (
            fallback_diagnostics.get(
                "away_recent_profile",
                {},
            )
        )


        home_games = (
            home_profile.get(
                "games",
                0,
            )
        )

        away_games = (
            away_profile.get(
                "games",
                0,
            )
        )


        if (
            home_games > 0
            and
            away_games > 0
        ):

            home_for = (
                home_profile.get(
                    "avg_pts_for",
                    0,
                )
            )

            away_for = (
                away_profile.get(
                    "avg_pts_for",
                    0,
                )
            )


            scoring_edge = (
                home_for
                -
                away_for
            )


            if abs(
                scoring_edge
            ) >= 4:

                leader = (
                    home
                    if scoring_edge > 0
                    else away
                )

                factors.append({
                    "label":
                        "Recent Scoring",

                    "detail":
                        (
                            f"{leader} has averaged "
                            f"{abs(scoring_edge):.1f} "
                            "more points per game across "
                            "its recent historical sample."
                        ),

                    "impact":
                        (
                            "high"
                            if abs(
                                scoring_edge
                            ) >= 8
                            else "medium"
                        ),
                })


            form_edge = (
                fallback_diagnostics.get(
                    "recent_form_edge",
                    0,
                )
            )


            if abs(
                form_edge
            ) >= 5:

                leader = (
                    home
                    if form_edge > 0
                    else away
                )

                factors.append({
                    "label":
                        "Recent Form",

                    "detail":
                        (
                            f"{leader} has the stronger "
                            "recent scoring-margin profile."
                        ),

                    "impact":
                        "medium",
                })


        else:

            missing_history = []

            if home_games <= 0:
                missing_history.append(
                    home
                )

            if away_games <= 0:
                missing_history.append(
                    away
                )


            if missing_history:

                factors.append({
                    "label":
                        "Limited Historical Sample",

                    "detail":
                        (
                            "Recent completed-game history "
                            "was not available for "
                            f"{', '.join(missing_history)}. "
                            "The fallback therefore uses "
                            "conservative scoring defaults."
                        ),

                    "impact":
                        "low",
                })


    # ========================================================
    # Home field
    # ========================================================

    if not features.get(
        "neutral_site",
        False,
    ):

        factors.append({
            "label":
                "Home Field",

            "detail":
                (
                    f"{home} gets "
                    "+3 pt HFA adjustment"
                ),

            "impact":
                "low",
        })


    return factors


# ============================================================
# Weekly Card Routes
# ============================================================

@app.get("/card/{league}")
async def weekly_card(
    league: str,
    week: int = 1,
    season: Optional[int] = None,
):

    if not app_state[
        "ready"
    ]:

        raise HTTPException(
            status_code=503,
            detail=(
                "Models are still initializing. "
                "Check /health and try again shortly."
            ),
        )


    if league.upper() not in (
        "NFL",
        "CFB",
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "League must be NFL or CFB"
            ),
        )


    try:

        return (
            await generate_weekly_card(
                league=
                    league.upper(),

                week=
                    week,

                season=
                    season,

                nfl_team_stats=
                    app_state[
                        "nfl_team_stats"
                    ],

                cfb_sp_lookup=
                    app_state[
                        "cfb_sp_lookup"
                    ],
            )
        )


    except Exception as exc:

        logger.exception(
            "Weekly card generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=str(
                exc
            ),
        )


# ============================================================
# Roster / Player Routes
# ============================================================

@app.get("/roster/status")
def roster_status():

    roster_engine.reload()

    return {
        "db_stats":
            roster_engine.get_db_stats(),

        "data_sources":
            get_data_source_status(),

        "position_groups":
            list(
                POSITION_GROUPS.keys()
            ),
    }


@app.get("/roster/team/{team_name}")
def team_profile(
    team_name: str,
):

    profile = (
        roster_engine.get_team_profile(
            team_name
        )
    )

    if not profile:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Team '{team_name}' "
                "not in roster DB."
            ),
        )

    return profile


@app.get("/roster/teams")
def all_teams():

    return (
        roster_engine.get_all_teams()
    )


@app.get("/roster/moves")
def recent_moves(
    limit: int = 50,
):

    return (
        roster_engine.get_recent_moves(
            limit=limit
        )
    )


@app.get("/roster/players")
def all_players(
    team: Optional[str] = None,
):

    return (
        roster_engine.get_all_players(
            team=team
        )
    )


@app.get("/roster/players/search")
def search_players(
    q: str,
):

    return (
        roster_engine.search_players(
            q
        )
    )


@app.post("/roster/player/add")
def add_player(
    req: AddPlayerRequest,
    secret: str = "",
):

    verify_admin(
        secret
    )

    player = (
        roster_engine.add_or_update_player(
            player_id=
                req.player_id,

            name=
                req.name,

            team=
                req.team,

            position_group=
                req.position_group,

            impact_score=
                req.impact_score,

            league=
                req.league,

            notes=
                req.notes,
        )
    )

    return {
        "player":
            player,

        "team_adjustment":
            roster_engine.get_team_adjustment(
                req.team
            ),
    }


@app.post("/roster/player/transfer")
def transfer_player(
    req: TransferPlayerRequest,
    secret: str = "",
):

    verify_admin(
        secret
    )

    try:

        return (
            roster_engine.transfer_player(
                player_id=
                    req.player_id,

                new_team=
                    req.new_team,

                move_type=
                    req.move_type,

                notes=
                    req.notes,
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(
                exc
            ),
        )


@app.post("/roster/sync")
async def sync_roster_moves(
    league: str = "NFL",
    secret: str = "",
):

    verify_admin(
        secret
    )

    return (
        await ingest_player_moves(
            league=league
        )
    )


# ============================================================
# Injury Routes
# ============================================================

@app.get("/injuries/{league}")
async def get_injuries(
    league: str,
):

    league = (
        league.upper()
    )

    all_injuries = (
        injury_engine.get_all_injuries(
            league=league
        )
    )

    summary = (
        injury_engine.get_status_summary(
            league=league
        )
    )

    return {
        "league":
            league,

        "by_team":
            all_injuries,

        "summary":
            summary,

        "last_updated":
            injury_engine.db.get(
                "last_updated"
            ),
    }


@app.post("/injuries/refresh")
async def refresh_injuries(
    league: str = "NFL",
    secret: str = "",
):

    verify_admin(
        secret
    )


    if league.upper() == "NFL":

        injuries = (
            await injury_engine.fetch_nfl_injuries()
        )

        injury_engine.update_injuries(
            injuries,
            "NFL",
        )


    else:

        if not app_state[
            "cfb_teams"
        ]:

            try:

                app_state[
                    "cfb_teams"
                ] = (
                    await get_cfb_teams()
                )

            except Exception as exc:

                raise HTTPException(
                    status_code=503,
                    detail=str(
                        exc
                    ),
                )


        team_ids = [
            team["id"]
            for team
            in app_state.get(
                "cfb_teams",
                [],
            )[:30]
            if team.get(
                "id"
            )
        ]


        injuries = (
            await injury_engine.fetch_cfb_injuries(
                team_ids
            )
        )

        injury_engine.update_injuries(
            injuries,
            "CFB",
        )


    return {
        "refreshed":
            len(
                injuries
            ),

        "league":
            league.upper(),
    }


@app.get("/injuries/team/{team_name}")
def team_injuries(
    team_name: str,
    league: str = "NFL",
):

    league = (
        league.upper()
    )

    injuries = (
        injury_engine.get_team_injuries(
            team_name,
            league,
        )
    )

    adjustment = (
        injury_engine.get_injury_adjustment(
            team_name,
            league,
            roster_engine,
        )
    )

    return {
        "team":
            team_name,

        "injuries":
            injuries,

        "rating_adjustment":
            adjustment[
                "adjustment"
            ],

        "depth_chart_cascades":
            adjustment[
                "depth_chart_cascades"
            ],
    }


# ============================================================
# Line Movement Routes
# ============================================================

@app.get("/movement/{league}")
async def line_movement(
    league: str,
    window_hours: int = 24,
):

    league = (
        league.upper()
    )

    movements = (
        snapshotter.get_all_movements(
            league,
            window_hours=
                window_hours,
        )
    )

    return {
        "league":
            league,

        "window_hours":
            window_hours,

        "games":
            movements,

        "snapshot_stats":
            snapshotter.get_snapshot_stats(),
    }


@app.get("/movement/game/{league}")
def game_movement(
    league: str,
    home_team: str,
    away_team: str,
    window_hours: int = 24,
):

    movement = (
        snapshotter.get_movement(
            home_team,
            away_team,
            window_hours=
                window_hours,
        )
    )

    return {
        "home_team":
            home_team,

        "away_team":
            away_team,

        **movement,
    }


@app.post("/movement/snapshot")
async def trigger_snapshot(
    league: str = "NFL",
    secret: str = "",
):

    verify_admin(
        secret
    )

    await snapshotter.take_snapshot(
        league.upper()
    )

    return {
        "ok":
            True,

        "stats":
            snapshotter.get_snapshot_stats(),
    }
