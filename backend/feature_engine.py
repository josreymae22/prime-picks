"""
feature_engine.py

Transforms raw NFL and CollegeFootballData game data into
model-ready features for Prime Picks.

Supports:
- ESPN NFL data
- Current CFBD camelCase responses
- Older cached snake_case CFBD responses
"""

import pandas as pd
import numpy as np
from typing import Optional


# ============================================================
# Helpers
# ============================================================

def _value(data: dict, camel: str, snake: str, default=None):
    """
    Read either current CFBD camelCase or legacy snake_case field.
    """
    value = data.get(camel)

    if value is None:
        value = data.get(snake)

    return default if value is None else value


# ============================================================
# NFL Feature Builder
# ============================================================

def build_nfl_game_features(
    games: list[dict],
) -> pd.DataFrame:
    """
    Build basic NFL training rows from completed games.
    """

    df = pd.DataFrame(games)

    if df.empty:
        return df

    df["margin"] = (
        df["home_score"]
        - df["away_score"]
    )

    df["total"] = (
        df["home_score"]
        + df["away_score"]
    )

    records = []

    for _, row in df.iterrows():

        records.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "margin": row["margin"],
            "total": row["total"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "season": row.get("season"),
            "week": row.get("week"),
        })

    return pd.DataFrame(records)


def build_nfl_team_rolling(
    games: list[dict],
    window: int = 6,
) -> dict:
    """
    Return rolling offensive/defensive averages keyed by team.
    """

    df = pd.DataFrame(games)

    if df.empty:
        return {}

    # Sort chronologically if possible.
    sort_cols = []

    if "season" in df.columns:
        sort_cols.append("season")

    if "week" in df.columns:
        sort_cols.append("week")

    if "date" in df.columns:
        sort_cols.append("date")

    if sort_cols:
        df = df.sort_values(
            sort_cols
        ).reset_index(
            drop=True
        )

    team_stats = {}

    all_teams = set(
        df["home_team"].tolist()
        + df["away_team"].tolist()
    )

    for team in all_teams:

        home_games = df[
            df["home_team"] == team
        ].copy()

        away_games = df[
            df["away_team"] == team
        ].copy()

        home_games["pts_for"] = (
            home_games["home_score"]
        )

        home_games["pts_against"] = (
            home_games["away_score"]
        )

        away_games["pts_for"] = (
            away_games["away_score"]
        )

        away_games["pts_against"] = (
            away_games["home_score"]
        )

        home_subset = home_games[
            ["pts_for", "pts_against"]
        ].copy()

        away_subset = away_games[
            ["pts_for", "pts_against"]
        ].copy()

        all_games = pd.concat(
            [
                home_subset,
                away_subset,
            ]
        ).sort_index()

        if all_games.empty:
            continue

        recent = all_games.tail(
            window
        )

        team_stats[team] = {
            "avg_pts_for":
                round(
                    recent["pts_for"].mean(),
                    2,
                ),

            "avg_pts_against":
                round(
                    recent["pts_against"].mean(),
                    2,
                ),

            "avg_margin":
                round(
                    (
                        recent["pts_for"]
                        - recent["pts_against"]
                    ).mean(),
                    2,
                ),
        }

    return team_stats


def build_nfl_matchup_features(
    home_team: str,
    away_team: str,
    team_stats: dict,
    neutral_site: bool = False,
) -> dict:
    """
    Generate NFL matchup features.
    """

    default_stats = {
        "avg_pts_for": 23.0,
        "avg_pts_against": 23.0,
        "avg_margin": 0.0,
    }

    home = team_stats.get(
        home_team,
        default_stats,
    )

    away = team_stats.get(
        away_team,
        default_stats,
    )

    hfa = (
        0.0
        if neutral_site
        else 2.5
    )

    return {
        "home_off_avg":
            home["avg_pts_for"],

        "home_def_avg":
            home["avg_pts_against"],

        "away_off_avg":
            away["avg_pts_for"],

        "away_def_avg":
            away["avg_pts_against"],

        "off_delta":
            home["avg_pts_for"]
            - away["avg_pts_for"],

        "def_delta":
            home["avg_pts_against"]
            - away["avg_pts_against"],

        "home_margin_avg":
            home["avg_margin"],

        "away_margin_avg":
            away["avg_margin"],

        "home_field_advantage":
            hfa,

        "combined_off":
            home["avg_pts_for"]
            + away["avg_pts_for"],

        "combined_def":
            home["avg_pts_against"]
            + away["avg_pts_against"],

        "neutral_site":
            neutral_site,
    }


# ============================================================
# CFB SP+ Feature Builder
# ============================================================

def build_cfb_sp_lookup(
    sp_ratings: list[dict],
) -> dict:
    """
    Build SP+ lookup keyed by team.

    Supports common CFBD SP+ response shapes.
    """

    lookup = {}

    for entry in sp_ratings:

        team = entry.get(
            "team",
            "",
        )

        if not team:
            continue

        offense = entry.get(
            "offense",
            {},
        )

        defense = entry.get(
            "defense",
            {},
        )

        if not isinstance(
            offense,
            dict,
        ):
            offense = {}

        if not isinstance(
            defense,
            dict,
        ):
            defense = {}

        lookup[team] = {
            "sp_overall":
                float(
                    entry.get(
                        "rating",
                        0.0,
                    )
                    or 0.0
                ),

            "sp_offense":
                float(
                    offense.get(
                        "rating",
                        0.0,
                    )
                    or 0.0
                ),

            "sp_defense":
                float(
                    defense.get(
                        "rating",
                        0.0,
                    )
                    or 0.0
                ),
        }

    return lookup


def build_cfb_matchup_features(
    home_team: str,
    away_team: str,
    sp_lookup: dict,
    neutral_site: bool = False,
) -> dict:
    """
    Generate CFB matchup features using SP+.
    """

    league_avg_sp = 0.0
    league_avg_off = 25.0
    league_avg_def = -5.0

    default_team = {
        "sp_overall":
            league_avg_sp,

        "sp_offense":
            league_avg_off,

        "sp_defense":
            league_avg_def,
    }

    home = sp_lookup.get(
        home_team,
        default_team,
    )

    away = sp_lookup.get(
        away_team,
        default_team,
    )

    hfa = (
        0.0
        if neutral_site
        else 3.0
    )

    return {
        "sp_diff":
            home["sp_overall"]
            - away["sp_overall"],

        "home_sp_overall":
            home["sp_overall"],

        "away_sp_overall":
            away["sp_overall"],

        "home_sp_offense":
            home["sp_offense"],

        "home_sp_defense":
            home["sp_defense"],

        "away_sp_offense":
            away["sp_offense"],

        "away_sp_defense":
            away["sp_defense"],

        "off_def_matchup_home":
            home["sp_offense"]
            - away["sp_defense"],

        "off_def_matchup_away":
            away["sp_offense"]
            - home["sp_defense"],

        "home_field_advantage":
            hfa,

        "predicted_home_off_contribution":
            home["sp_offense"]
            + hfa,

        "predicted_away_off_contribution":
            away["sp_offense"],

        "neutral_site":
            neutral_site,
    }


# ============================================================
# CFB Training Data Builder
# ============================================================

def build_cfb_training_data(
    games: list[dict],
    sp_lookup: dict,
) -> pd.DataFrame:
    """
    Convert CFBD historical games into model-training rows.

    Handles BOTH:

        Current CFBD:
            homeTeam
            awayTeam
            homePoints
            awayPoints
            neutralSite

        Legacy/cache:
            home_team
            away_team
            home_points
            away_points
            neutral_site
    """

    rows = []

    skipped_missing_score = 0
    skipped_missing_team = 0

    for game in games:

        home_points = _value(
            game,
            "homePoints",
            "home_points",
        )

        away_points = _value(
            game,
            "awayPoints",
            "away_points",
        )

        if (
            home_points is None
            or away_points is None
        ):
            skipped_missing_score += 1
            continue

        home_team = _value(
            game,
            "homeTeam",
            "home_team",
            "",
        )

        away_team = _value(
            game,
            "awayTeam",
            "away_team",
            "",
        )

        if (
            not home_team
            or not away_team
        ):
            skipped_missing_team += 1
            continue

        neutral_site = bool(
            _value(
                game,
                "neutralSite",
                "neutral_site",
                False,
            )
        )

        try:
            home_points = float(
                home_points
            )

            away_points = float(
                away_points
            )

        except (
            TypeError,
            ValueError,
        ):
            skipped_missing_score += 1
            continue

        features = (
            build_cfb_matchup_features(
                home_team,
                away_team,
                sp_lookup,
                neutral_site=
                    neutral_site,
            )
        )

        features["margin"] = (
            home_points
            - away_points
        )

        features["total"] = (
            home_points
            + away_points
        )

        features["home_score"] = (
            home_points
        )

        features["away_score"] = (
            away_points
        )

        # Useful diagnostics.
        features["season"] = _value(
            game,
            "season",
            "season",
        )

        features["week"] = _value(
            game,
            "week",
            "week",
        )

        features["home_team"] = (
            home_team
        )

        features["away_team"] = (
            away_team
        )

        rows.append(
            features
        )

    df = pd.DataFrame(
        rows
    )

    # Store diagnostics without interfering with sklearn features.
    df.attrs[
        "skipped_missing_score"
    ] = skipped_missing_score

    df.attrs[
        "skipped_missing_team"
    ] = skipped_missing_team

    return df
