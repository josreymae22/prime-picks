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
- Both teams have SP+ -> trained CFB model
- Missing SP+ for either team -> opponent-adjusted historical fallback

The fallback uses:
- Recent scoring offense
- Recent scoring defense
- Recent scoring margin
- Strength of schedule
- Iterative SRS-style opponent-adjusted power rating
- Home-field advantage

This prevents FCS / weaker-schedule teams from being treated as equal
to FBS teams merely because their raw scoring averages look similar.
"""

import asyncio
import logging
import math
import os

from datetime import datetime, timezone
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
from database import AsyncSessionLocal, init_db
from social_autopost import (
    Game as SocialGame,
    run_poller,
    schedule_slate_posts,
)
from slate_card import Pick


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
    "autopost_task": None,
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
    """
    Read either current CFBD camelCase fields or older
    snake_case fields.
    """

    value = data.get(camel)

    if value is None:
        value = data.get(snake)

    return default if value is None else value


def _clip(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


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


# ============================================================
# CFB opponent-adjusted fallback profiles
# ============================================================

def build_cfb_recent_team_stats(
    games: list[dict],
    window: int = 12,
    srs_iterations: int = 30,
) -> dict:
    """
    Build opponent-adjusted CFB fallback team profiles.

    This replaces the older raw-scoring-only fallback.

    For every team we retain recent:
    - Points scored
    - Points allowed
    - Scoring margin
    - Opponents

    Then calculate an iterative SRS-style rating:

        team rating
        =
        average (
            game scoring margin
            +
            opponent rating
        )

    The ratings are re-centered around zero each iteration.

    Why this matters:
    A team averaging 30 PPG against weak opposition should not
    automatically be considered equivalent to a team averaging
    30 PPG against SEC / Big Ten / Big 12 competition.

    Large individual-game margins are capped during SRS
    calculation to reduce the effect of extreme blowouts.
    """

    sortable_games = []

    # --------------------------------------------------------
    # Normalize historical games
    # --------------------------------------------------------

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
        key=lambda game: (
            game["season"],
            game["week"],
            game["date"],
        )
    )

    # --------------------------------------------------------
    # Team histories
    # --------------------------------------------------------

    team_history: dict[str, list[dict]] = {}

    for game in sortable_games:

        home_team = game["home_team"]
        away_team = game["away_team"]

        home_points = game["home_points"]
        away_points = game["away_points"]

        home_margin = (
            home_points
            -
            away_points
        )

        away_margin = (
            away_points
            -
            home_points
        )

        team_history.setdefault(
            home_team,
            [],
        ).append({
            "opponent": away_team,
            "pts_for": home_points,
            "pts_against": away_points,
            "margin": home_margin,
        })

        team_history.setdefault(
            away_team,
            [],
        ).append({
            "opponent": home_team,
            "pts_for": away_points,
            "pts_against": home_points,
            "margin": away_margin,
        })

    # --------------------------------------------------------
    # Restrict each team to recent history
    # --------------------------------------------------------

    recent_history = {
        team: history[-window:]
        for team, history
        in team_history.items()
        if history
    }

    # --------------------------------------------------------
    # Initial ratings = recent average margin
    # --------------------------------------------------------

    ratings = {}

    for team, history in recent_history.items():

        margins = [
            _clip(
                float(game["margin"]),
                -35.0,
                35.0,
            )
            for game in history
        ]

        ratings[team] = (
            float(
                np.mean(margins)
            )
            if margins
            else 0.0
        )

    # --------------------------------------------------------
    # Iterative Simple Rating System
    # --------------------------------------------------------

    for _ in range(
        srs_iterations
    ):

        new_ratings = {}

        for team, history in recent_history.items():

            game_values = []

            for game in history:

                opponent = (
                    game[
                        "opponent"
                    ]
                )

                margin = _clip(
                    float(
                        game[
                            "margin"
                        ]
                    ),
                    -35.0,
                    35.0,
                )

                opponent_rating = (
                    ratings.get(
                        opponent,
                        0.0,
                    )
                )

                game_values.append(
                    margin
                    +
                    opponent_rating
                )

            if game_values:

                new_ratings[
                    team
                ] = float(
                    np.mean(
                        game_values
                    )
                )

            else:

                new_ratings[
                    team
                ] = 0.0

        # Center ratings around zero.
        if new_ratings:

            center = float(
                np.mean(
                    list(
                        new_ratings.values()
                    )
                )
            )

            for team in new_ratings:

                new_ratings[
                    team
                ] -= center

        # Damp movement slightly for stability.
        for team in new_ratings:

            previous = ratings.get(
                team,
                0.0,
            )

            new_ratings[
                team
            ] = (
                0.75
                *
                new_ratings[
                    team
                ]
                +
                0.25
                *
                previous
            )

        ratings = new_ratings

    # --------------------------------------------------------
    # Build final profiles
    # --------------------------------------------------------

    profiles = {}

    for team, history in recent_history.items():

        pts_for = [
            float(
                game[
                    "pts_for"
                ]
            )
            for game in history
        ]

        pts_against = [
            float(
                game[
                    "pts_against"
                ]
            )
            for game in history
        ]

        margins = [
            float(
                game[
                    "margin"
                ]
            )
            for game in history
        ]

        opponent_ratings = [
            ratings.get(
                game[
                    "opponent"
                ],
                0.0,
            )
            for game in history
        ]

        avg_pts_for = float(
            np.mean(
                pts_for
            )
        )

        avg_pts_against = float(
            np.mean(
                pts_against
            )
        )

        avg_margin = float(
            np.mean(
                margins
            )
        )

        schedule_strength = (
            float(
                np.mean(
                    opponent_ratings
                )
            )
            if opponent_ratings
            else 0.0
        )

        power_rating = float(
            ratings.get(
                team,
                0.0,
            )
        )

        profiles[
            team
        ] = {
            "avg_pts_for":
                round(
                    avg_pts_for,
                    2,
                ),

            "avg_pts_against":
                round(
                    avg_pts_against,
                    2,
                ),

            "avg_margin":
                round(
                    avg_margin,
                    2,
                ),

            "schedule_strength":
                round(
                    schedule_strength,
                    2,
                ),

            "power_rating":
                round(
                    power_rating,
                    2,
                ),

            "srs_rating":
                round(
                    power_rating,
                    2,
                ),

            "games":
                len(
                    history
                ),
        }

    return profiles


# ============================================================
# CFB opponent-adjusted fallback prediction
# ============================================================

def predict_cfb_fallback(
    home_team: str,
    away_team: str,
    team_stats: dict,
    neutral_site: bool = False,
) -> tuple[dict, dict]:
    """
    Opponent-adjusted CFB fallback.

    Used only when one or both teams do not have valid SP+ data.

    Combines:

    1. Raw scoring projection
       team's offense vs opponent defense

    2. Opponent-adjusted SRS power difference

    3. Home-field advantage

    SRS receives more weight than raw scoring because raw scoring
    can be highly misleading across different levels of competition.
    """

    default_profile = {
        "avg_pts_for": 27.0,
        "avg_pts_against": 27.0,
        "avg_margin": 0.0,

        "schedule_strength": 0.0,
        "power_rating": 0.0,
        "srs_rating": 0.0,

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
    # Raw scoring projection
    # --------------------------------------------------------

    raw_home_score = (
        (
            home[
                "avg_pts_for"
            ]
            +
            away[
                "avg_pts_against"
            ]
        )
        /
        2.0
    )

    raw_away_score = (
        (
            away[
                "avg_pts_for"
            ]
            +
            home[
                "avg_pts_against"
            ]
        )
        /
        2.0
    )

    raw_total = (
        raw_home_score
        +
        raw_away_score
    )

    raw_scoring_margin = (
        raw_home_score
        -
        raw_away_score
    )

    # --------------------------------------------------------
    # Opponent-adjusted strength
    # --------------------------------------------------------

    home_power = float(
        home.get(
            "power_rating",
            0.0,
        )
        or 0.0
    )

    away_power = float(
        away.get(
            "power_rating",
            0.0,
        )
        or 0.0
    )

    power_edge = (
        home_power
        -
        away_power
    )

    # SRS difference approximates expected neutral-field margin.
    power_margin = (
        power_edge
        +
        hfa
    )

    # --------------------------------------------------------
    # Blend raw scoring and SRS
    #
    # Opponent-adjusted rating receives most of the margin weight.
    # --------------------------------------------------------

    predicted_margin = (
        0.25
        *
        (
            raw_scoring_margin
            +
            hfa
        )
        +
        0.75
        *
        power_margin
    )

    # --------------------------------------------------------
    # Total
    #
    # Raw scoring information is still useful for totals.
    # --------------------------------------------------------

    predicted_total = _clip(
        raw_total,
        34.0,
        85.0,
    )

    # Avoid absurd margins.
    predicted_margin = _clip(
        predicted_margin,
        -45.0,
        45.0,
    )

    # A margin cannot realistically exceed almost the entire total.
    max_margin_from_total = max(
        10.0,
        predicted_total
        -
        10.0,
    )

    predicted_margin = _clip(
        predicted_margin,
        -max_margin_from_total,
        max_margin_from_total,
    )

    # --------------------------------------------------------
    # Convert margin + total to team scores
    # --------------------------------------------------------

    home_score = (
        predicted_total
        +
        predicted_margin
    ) / 2.0

    away_score = (
        predicted_total
        -
        predicted_margin
    ) / 2.0

    home_score = max(
        3.0,
        home_score,
    )

    away_score = max(
        3.0,
        away_score,
    )

    # Recalculate after score floor.
    predicted_total = (
        home_score
        +
        away_score
    )

    predicted_margin = (
        home_score
        -
        away_score
    )

    # --------------------------------------------------------
    # Wider uncertainty than trained SP+ model
    # --------------------------------------------------------

    margin_rmse = 16.0
    total_rmse = 19.0

    margin_lo = (
        predicted_margin
        -
        1.28
        *
        margin_rmse
    )

    margin_hi = (
        predicted_margin
        +
        1.28
        *
        margin_rmse
    )

    total_lo = (
        predicted_total
        -
        1.28
        *
        total_rmse
    )

    total_hi = (
        predicted_total
        +
        1.28
        *
        total_rmse
    )

    home_win_prob = _normal_cdf(
        predicted_margin
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
                predicted_margin,
                1,
            ),

        "predicted_total":
            round(
                predicted_total,
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
                float(
                    home_win_prob
                ),
                3,
            ),

        "model_trained":
            False,

        "prediction_mode":
            "opponent_adjusted_fallback",
    }

    diagnostics = {
        "home_recent_profile":
            home,

        "away_recent_profile":
            away,

        "home_field_advantage":
            hfa,

        "home_power_rating":
            round(
                home_power,
                2,
            ),

        "away_power_rating":
            round(
                away_power,
                2,
            ),

        "power_rating_edge":
            round(
                power_edge,
                2,
            ),

        "power_margin":
            round(
                power_margin,
                2,
            ),

        "raw_scoring_margin":
            round(
                raw_scoring_margin,
                2,
            ),

        "raw_projected_total":
            round(
                raw_total,
                2,
            ),

        "home_schedule_strength":
            round(
                float(
                    home.get(
                        "schedule_strength",
                        0.0,
                    )
                    or 0.0
                ),
                2,
            ),

        "away_schedule_strength":
            round(
                float(
                    away.get(
                        "schedule_strength",
                        0.0,
                    )
                    or 0.0
                ),
                2,
            ),

        "home_history_available":
            home.get(
                "games",
                0,
            )
            > 0,

        "away_history_available":
            away.get(
                "games",
                0,
            )
            > 0,

        "fallback_method":
            "opponent_adjusted_srs",
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

        actual_team_count = len([
            key
            for key
            in lookup.keys()
            if not str(
                key
            ).startswith(
                "__"
            )
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
            training_seasons[
                -1
            ]
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
                for key
                in lookup.keys()
                if not str(
                    key
                ).startswith(
                    "__"
                )
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

    Runs in the background AFTER FastAPI binds to Render's port.
    """

    app_state[
        "initializing"
    ] = True

    app_state[
        "ready"
    ] = False

    app_state[
        "startup_error"
    ] = None

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

        app_state[
            "nfl_teams"
        ] = (
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
                seasons=
                    nfl_seasons
            )
        )

        logger.info(
            "NFL historical games loaded: %s",
            len(
                nfl_games
            ),
        )

        # ====================================================
        # NFL TEAM FORM
        # ====================================================

        app_state[
            "nfl_team_stats"
        ] = (
            build_nfl_team_rolling(
                nfl_games,
                window=8,
            )
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

        if len(
            nfl_games
        ) > 20:

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
                len(
                    nfl_train_df
                ),
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
                seasons=
                    cfb_seasons
            )
        )

        logger.info(
            "CFB historical games loaded: %s",
            len(
                cfb_games
            ),
        )

        # ====================================================
        # CFB OPPONENT-ADJUSTED FALLBACK PROFILES
        # ====================================================

        logger.info(
            "Building opponent-adjusted CFB fallback profiles..."
        )

        app_state[
            "cfb_team_stats"
        ] = (
            build_cfb_recent_team_stats(
                cfb_games,
                window=12,
                srs_iterations=30,
            )
        )

        logger.info(
            "CFB opponent-adjusted profiles loaded: %s",
            len(
                app_state[
                    "cfb_team_stats"
                ]
            ),
        )

        # Diagnostic extremes
        if app_state[
            "cfb_team_stats"
        ]:

            sorted_power = sorted(
                app_state[
                    "cfb_team_stats"
                ].items(),
                key=lambda item:
                    item[
                        1
                    ].get(
                        "power_rating",
                        0,
                    ),
                reverse=True,
            )

            logger.info(
                "CFB highest fallback power ratings: %s",
                [
                    (
                        team,
                        profile.get(
                            "power_rating"
                        ),
                    )
                    for team, profile
                    in sorted_power[
                        :5
                    ]
                ],
            )

            logger.info(
                "CFB lowest fallback power ratings: %s",
                [
                    (
                        team,
                        profile.get(
                            "power_rating"
                        ),
                    )
                    for team, profile
                    in sorted_power[
                        -5:
                    ]
                ],
            )

        # ====================================================
        # CFB SEASON-SPECIFIC SP+ TRAINING
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
                for game
                in cfb_games
                if game.get(
                    "season"
                )
                == season
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
        # COMBINE CFB TRAINING
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
        ] = (
            cfb_training_diagnostics
        )

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
                    len(
                        players
                    )
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
            "CFB opponent-adjusted fallback profiles: %s",
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
# Social auto-post background service
# ============================================================

async def initialize_social_autopost():
    """
    Initialize the social-post database and start the background poller.

    This runs as a background task so Render can bind the web service
    immediately instead of waiting for PostgreSQL setup during startup.
    """

    try:

        logger.info(
            "Initializing social autopost database..."
        )

        await init_db()

        logger.info(
            "Social autopost database ready."
        )

        await run_poller(
            AsyncSessionLocal
        )

    except asyncio.CancelledError:

        logger.info(
            "Social autopost task cancelled."
        )

        raise

    except Exception as exc:

        logger.error(
            "Social autopost startup/runtime error: %s",
            exc,
            exc_info=True,
        )


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

    # Start social database + poller in the background.
    # This intentionally does not block Render startup.
    autopost_task = (
        asyncio.create_task(
            initialize_social_autopost()
        )
    )

    app_state[
        "autopost_task"
    ] = autopost_task

    # Let Render see the port immediately.
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

    autopost_task = (
        app_state.get(
            "autopost_task"
        )
    )

    if autopost_task:

        autopost_task.cancel()

        try:

            await autopost_task

        except asyncio.CancelledError:

            pass

        except Exception as exc:

            logger.warning(
                "Social autopost shutdown error: %s",
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
    version="1.4.0",
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

    power_values = [
        profile.get(
            "power_rating",
            0.0,
        )
        for profile
        in app_state[
            "cfb_team_stats"
        ].values()
    ]

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

        "cfb_fallback_method":
            "opponent_adjusted_srs",

        "cfb_power_rating_min":
            (
                round(
                    min(
                        power_values
                    ),
                    2,
                )
                if power_values
                else None
            ),

        "cfb_power_rating_max":
            (
                round(
                    max(
                        power_values
                    ),
                    2,
                )
                if power_values
                else None
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

        fallback_diagnostics = None

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

        # ----------------------------------------------------
        # Both teams have SP+
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Missing SP+ -> opponent-adjusted SRS fallback
        # ----------------------------------------------------

        else:

            (
                prediction,
                fallback_diagnostics,
            ) = (
                predict_cfb_fallback(
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
            )

            prediction_mode = (
                "opponent_adjusted_fallback"
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
        fallback_diagnostics
        is not None
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

    off_delta = (
        features.get(
            "off_delta",
            0,
        )
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
                    "more pts/game recently"
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

    def_delta = (
        features.get(
            "def_delta",
            0,
        )
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
                    "fewer points on average"
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
                    "+2.5 pt HFA adjustment"
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
                    "significantly better "
                    "recent win margins"
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

    # --------------------------------------------------------
    # Missing SP+
    # --------------------------------------------------------

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
                        "Prime Picks used opponent-adjusted "
                        "historical ratings instead."
                    ),

                "impact":
                    "low",
            })

    # --------------------------------------------------------
    # Normal SP+
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Opponent-adjusted fallback factors
    # --------------------------------------------------------

    elif fallback_diagnostics:

        home_power = float(
            fallback_diagnostics.get(
                "home_power_rating",
                0.0,
            )
            or 0.0
        )

        away_power = float(
            fallback_diagnostics.get(
                "away_power_rating",
                0.0,
            )
            or 0.0
        )

        power_edge = (
            home_power
            -
            away_power
        )

        if abs(
            power_edge
        ) >= 4.0:

            leader = (
                home
                if power_edge > 0
                else away
            )

            factors.append({
                "label":
                    "Opponent-Adjusted Strength",

                "detail":
                    (
                        f"{leader} has a "
                        f"{abs(power_edge):.1f} pt "
                        "SRS-style power-rating advantage."
                    ),

                "impact":
                    (
                        "high"
                        if abs(
                            power_edge
                        ) >= 15
                        else "medium"
                    ),
            })

        home_sos = float(
            fallback_diagnostics.get(
                "home_schedule_strength",
                0.0,
            )
            or 0.0
        )

        away_sos = float(
            fallback_diagnostics.get(
                "away_schedule_strength",
                0.0,
            )
            or 0.0
        )

        sos_edge = (
            home_sos
            -
            away_sos
        )

        if abs(
            sos_edge
        ) >= 4.0:

            leader = (
                home
                if sos_edge > 0
                else away
            )

            factors.append({
                "label":
                    "Schedule Strength",

                "detail":
                    (
                        f"{leader} has faced the "
                        "stronger recent opponent profile."
                    ),

                "impact":
                    "medium",
            })

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
            home_games <= 0
            or
            away_games <= 0
        ):

            missing_history = []

            if home_games <= 0:
                missing_history.append(
                    home
                )

            if away_games <= 0:
                missing_history.append(
                    away
                )

            factors.append({
                "label":
                    "Limited Historical Sample",

                "detail":
                    (
                        "Recent completed-game history "
                        "was not available for "
                        f"{', '.join(missing_history)}."
                    ),

                "impact":
                    "low",
            })

    # --------------------------------------------------------
    # Home field
    # --------------------------------------------------------

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
# Weekly Card -> Social Auto-Post bridge
# ============================================================

def _parse_social_kickoff(
    value,
) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(
                raw.replace("Z", "+00:00")
            )
        except ValueError:
            logger.warning(
                "Social queue skipped invalid kickoff: %r",
                value,
            )
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _format_social_line(
    value: float,
) -> str:
    rounded = round(float(value), 1)

    if rounded == 0:
        return "PK"

    number = (
        f"{rounded:.1f}"
        .rstrip("0")
        .rstrip(".")
    )

    if rounded > 0:
        return f"+{number}"

    return number


def _social_pick_from_card_game(
    game: dict,
) -> Optional[Pick]:
    disparity = (
        game.get("disparity", {})
        or {}
    )

    edge_label = (
        disparity.get("edge_label", "")
        or ""
    ).lower()

    if "strong edge" not in edge_label:
        return None

    home = (
        game.get("home_team", "")
        or ""
    )
    away = (
        game.get("away_team", "")
        or ""
    )

    if not home or not away:
        return None

    spread_type = disparity.get(
        "spread_edge_type"
    )
    total_type = disparity.get(
        "total_edge_type"
    )
    vegas_spread = disparity.get(
        "vegas_spread"
    )
    vegas_total = disparity.get(
        "vegas_total"
    )
    spread_disparity = disparity.get(
        "spread_disparity"
    )
    total_disparity = disparity.get(
        "total_disparity"
    )

    try:
        spread_strength = (
            abs(float(spread_disparity or 0.0))
            * 3.0
        )
    except (TypeError, ValueError):
        spread_strength = 0.0

    try:
        total_strength = (
            abs(float(total_disparity or 0.0))
            * 1.5
        )
    except (TypeError, ValueError):
        total_strength = 0.0

    spread_pick_available = (
        spread_type
        in ("lean_home", "lean_away")
        and vegas_spread is not None
    )

    total_pick_available = (
        total_type
        in ("lean_over", "lean_under")
        and vegas_total is not None
    )

    selection = None

    if (
        spread_pick_available
        and (
            spread_strength >= total_strength
            or not total_pick_available
        )
    ):
        try:
            home_spread = float(vegas_spread)
        except (TypeError, ValueError):
            return None

        if spread_type == "lean_home":
            selection = (
                f"{home} "
                f"{_format_social_line(home_spread)}"
            )
        else:
            selection = (
                f"{away} "
                f"{_format_social_line(-home_spread)}"
            )

    elif total_pick_available:
        try:
            total_number = float(vegas_total)
        except (TypeError, ValueError):
            return None

        total_text = (
            f"{total_number:.1f}"
            .rstrip("0")
            .rstrip(".")
        )

        if total_type == "lean_over":
            selection = f"Over {total_text}"
        else:
            selection = f"Under {total_text}"

    if not selection:
        logger.info(
            "Strong Edge social game skipped because no actionable "
            "spread/total selection was available: %s @ %s",
            away,
            home,
        )
        return None

    return Pick(
        away=away,
        home=home,
        selection=selection,
        price=-110,
        units=1.0,
    )


def _social_games_from_weekly_card(
    card: dict,
) -> list[SocialGame]:
    social_games: list[SocialGame] = []

    for game in (
        card.get("games", [])
        or []
    ):
        disparity = (
            game.get("disparity", {})
            or {}
        )

        edge_label = (
            disparity.get("edge_label", "")
            or ""
        )

        if "strong edge" not in edge_label.lower():
            continue

        kickoff = _parse_social_kickoff(
            game.get("date")
        )

        if kickoff is None:
            continue

        pick = _social_pick_from_card_game(
            game
        )

        if pick is None:
            continue

        game_id = str(
            game.get("game_id")
            or (
                f"{game.get('away_team', '')}"
                f"-{game.get('home_team', '')}"
                f"-{game.get('date', '')}"
            )
        )

        social_games.append(
            SocialGame(
                game_id=game_id,
                commence_time=kickoff,
                pick=pick,
                edge_label=edge_label,
                strong_edge=True,
                confidence=(
                    game.get("confidence", "")
                    or ""
                ),
                prediction_mode=(
                    game.get("prediction_mode", "")
                    or ""
                ),
                sharp_signal=float(
                    disparity.get("sharp_signal", 0.0)
                    or 0.0
                ),
                steam_move=bool(
                    disparity.get("steam_move", False)
                ),
            )
        )

    return social_games


async def _schedule_social_from_weekly_card(
    card: dict,
) -> int:
    social_games = (
        _social_games_from_weekly_card(card)
    )

    if not social_games:
        logger.info(
            "Weekly Card produced no Strong Edge games eligible "
            "for social queue conversion."
        )
        return 0

    sport = (
        card.get("league", "")
        or ""
    ).upper()

    week = int(
        card.get("week", 1)
        or 1
    )

    async with AsyncSessionLocal() as session:
        rows = await schedule_slate_posts(
            session,
            sport=sport,
            week_label=f"Week {week}",
            games=social_games,
        )

    logger.info(
        "Weekly Card -> social queue complete: %s row(s) returned "
        "for %s Week %s.",
        len(rows),
        sport,
        week,
    )

    return len(rows)


# ============================================================
# Weekly Card
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

        card = (
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

                cfb_team_stats=
                    app_state[
                        "cfb_team_stats"
                    ],
            )
        )

        try:
            await _schedule_social_from_weekly_card(
                card
            )

        except Exception as social_exc:
            logger.exception(
                "Weekly Card social scheduling failed (non-fatal): %s",
                social_exc,
            )

        return card

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
            team[
                "id"
            ]
            for team
            in app_state.get(
                "cfb_teams",
                [],
            )[
                :30
            ]
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
