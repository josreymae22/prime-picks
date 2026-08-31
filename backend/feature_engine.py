"""
feature_engine.py

Transforms raw NFL and CollegeFootballData game data into
model-ready features for Prime Picks.

Supports:
- ESPN NFL data
- Current CFBD camelCase responses
- Older cached snake_case CFBD responses

CFB safety:
- Tracks whether SP+ exists for each team
- Does NOT create fake SP+ advantages for missing teams
- Skips historical training rows when either team has no SP+ rating
"""

import pandas as pd
import numpy as np


# ============================================================
# Helpers
# ============================================================

def _value(
    data: dict,
    camel: str,
    snake: str,
    default=None,
):
    """
    Read either current CFBD camelCase or legacy snake_case field.
    """

    value = data.get(camel)

    if value is None:
        value = data.get(snake)

    return default if value is None else value


def _normalize_team_name(name: str) -> str:
    """
    Normalize a team name for fallback comparisons.

    IMPORTANT:
    We still preserve the original CFBD/SP+ team name as
    the primary lookup key.
    """

    if not name:
        return ""

    return (
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("_", " ")
    )


# ============================================================
# NFL Feature Builder
# ============================================================

def build_nfl_game_features(
    games: list[dict],
) -> pd.DataFrame:

    df = pd.DataFrame(games)

    if df.empty:
        return df

    df["margin"] = (
        df["home_score"]
        -
        df["away_score"]
    )

    df["total"] = (
        df["home_score"]
        +
        df["away_score"]
    )

    records = []

    for _, row in df.iterrows():

        records.append({
            "home_team":
                row["home_team"],

            "away_team":
                row["away_team"],

            "margin":
                row["margin"],

            "total":
                row["total"],

            "home_score":
                row["home_score"],

            "away_score":
                row["away_score"],

            "season":
                row.get("season"),

            "week":
                row.get("week"),
        })

    return pd.DataFrame(records)


def build_nfl_team_rolling(
    games: list[dict],
    window: int = 6,
) -> dict:

    df = pd.DataFrame(games)

    if df.empty:
        return {}

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
        +
        df["away_team"].tolist()
    )


    for team in all_teams:

        home_games = df[
            df["home_team"] == team
        ].copy()

        away_games = df[
            df["away_team"] == team
        ].copy()


        home_games[
            "pts_for"
        ] = home_games[
            "home_score"
        ]

        home_games[
            "pts_against"
        ] = home_games[
            "away_score"
        ]


        away_games[
            "pts_for"
        ] = away_games[
            "away_score"
        ]

        away_games[
            "pts_against"
        ] = away_games[
            "home_score"
        ]


        home_subset = home_games[
            [
                "pts_for",
                "pts_against",
            ]
        ].copy()

        away_subset = away_games[
            [
                "pts_for",
                "pts_against",
            ]
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
                    recent[
                        "pts_for"
                    ].mean(),
                    2,
                ),

            "avg_pts_against":
                round(
                    recent[
                        "pts_against"
                    ].mean(),
                    2,
                ),

            "avg_margin":
                round(
                    (
                        recent[
                            "pts_for"
                        ]
                        -
                        recent[
                            "pts_against"
                        ]
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

    default_stats = {
        "avg_pts_for":
            23.0,

        "avg_pts_against":
            23.0,

        "avg_margin":
            0.0,
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
            home[
                "avg_pts_for"
            ],

        "home_def_avg":
            home[
                "avg_pts_against"
            ],

        "away_off_avg":
            away[
                "avg_pts_for"
            ],

        "away_def_avg":
            away[
                "avg_pts_against"
            ],

        "off_delta":
            home[
                "avg_pts_for"
            ]
            -
            away[
                "avg_pts_for"
            ],

        "def_delta":
            home[
                "avg_pts_against"
            ]
            -
            away[
                "avg_pts_against"
            ],

        "home_margin_avg":
            home[
                "avg_margin"
            ],

        "away_margin_avg":
            away[
                "avg_margin"
            ],

        "home_field_advantage":
            hfa,

        "combined_off":
            home[
                "avg_pts_for"
            ]
            +
            away[
                "avg_pts_for"
            ],

        "combined_def":
            home[
                "avg_pts_against"
            ]
            +
            away[
                "avg_pts_against"
            ],

        "neutral_site":
            neutral_site,
    }


def build_nfl_training_data(
    games: list[dict],
    window: int = 8,
) -> pd.DataFrame:

    if not games:
        return pd.DataFrame()


    games = sorted(
        games,
        key=lambda g: (
            g.get(
                "season",
                0,
            ),
            g.get(
                "week",
                0,
            ),
            g.get(
                "date",
                "",
            ),
        ),
    )


    team_history = {}
    rows = []


    def recent_stats(
        team: str,
    ):

        history = (
            team_history.get(
                team,
                [],
            )[-window:]
        )


        if not history:

            return {
                "avg_pts_for":
                    23.0,

                "avg_pts_against":
                    23.0,

                "avg_margin":
                    0.0,
            }


        pts_for = [
            g["pts_for"]
            for g in history
        ]

        pts_against = [
            g["pts_against"]
            for g in history
        ]


        return {

            "avg_pts_for":
                float(
                    np.mean(
                        pts_for
                    )
                ),

            "avg_pts_against":
                float(
                    np.mean(
                        pts_against
                    )
                ),

            "avg_margin":
                float(
                    np.mean(
                        [
                            pf - pa
                            for pf, pa
                            in zip(
                                pts_for,
                                pts_against,
                            )
                        ]
                    )
                ),
        }


    for game in games:

        home_team = game[
            "home_team"
        ]

        away_team = game[
            "away_team"
        ]


        stats = {
            home_team:
                recent_stats(
                    home_team
                ),

            away_team:
                recent_stats(
                    away_team
                ),
        }


        features = (
            build_nfl_matchup_features(
                home_team,
                away_team,
                stats,
                neutral_site=False,
            )
        )


        home_score = float(
            game[
                "home_score"
            ]
        )

        away_score = float(
            game[
                "away_score"
            ]
        )


        features[
            "margin"
        ] = (
            home_score
            -
            away_score
        )

        features[
            "total"
        ] = (
            home_score
            +
            away_score
        )

        features[
            "season"
        ] = game.get(
            "season"
        )

        features[
            "week"
        ] = game.get(
            "week"
        )

        features[
            "home_team"
        ] = home_team

        features[
            "away_team"
        ] = away_team


        rows.append(
            features
        )


        team_history.setdefault(
            home_team,
            [],
        ).append({
            "pts_for":
                home_score,

            "pts_against":
                away_score,
        })


        team_history.setdefault(
            away_team,
            [],
        ).append({
            "pts_for":
                away_score,

            "pts_against":
                home_score,
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# CFB SP+ Lookup
# ============================================================

def build_cfb_sp_lookup(
    sp_ratings: list[dict],
) -> dict:
    """
    Build SP+ lookup keyed by exact CFBD team name.

    Also stores normalized aliases for safer matching.

    Missing teams are NOT assigned fake ratings.
    """

    lookup = {}
    normalized_lookup = {}


    for entry in sp_ratings:

        team = (
            entry.get(
                "team",
                ""
            )
            or ""
        ).strip()


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


        rating = float(
            entry.get(
                "rating",
                0.0,
            )
            or 0.0
        )


        offense_rating = float(
            offense.get(
                "rating",
                0.0,
            )
            or 0.0
        )


        defense_rating = float(
            defense.get(
                "rating",
                0.0,
            )
            or 0.0
        )


        record = {

            "team":
                team,

            "sp_overall":
                rating,

            "sp_offense":
                offense_rating,

            "sp_defense":
                defense_rating,

            "has_sp_data":
                True,
        }


        lookup[
            team
        ] = record


        normalized_lookup[
            _normalize_team_name(
                team
            )
        ] = team


    lookup[
        "__normalized__"
    ] = normalized_lookup


    return lookup


def _get_cfb_sp_team(
    team_name: str,
    sp_lookup: dict,
):
    """
    Return SP+ record for a team.

    Exact matching is attempted first.
    Then normalized exact-name matching.

    We intentionally do NOT do fuzzy matching because
    Georgia != Georgia State and North Carolina !=
    North Carolina A&T.
    """

    if not team_name:
        return None


    if team_name in sp_lookup:

        return sp_lookup[
            team_name
        ]


    normalized = (
        _normalize_team_name(
            team_name
        )
    )


    normalized_lookup = (
        sp_lookup.get(
            "__normalized__",
            {},
        )
    )


    matched_key = (
        normalized_lookup.get(
            normalized
        )
    )


    if (
        matched_key
        and matched_key
        in sp_lookup
    ):

        return sp_lookup[
            matched_key
        ]


    return None


# ============================================================
# CFB Matchup Features
# ============================================================

def build_cfb_matchup_features(
    home_team: str,
    away_team: str,
    sp_lookup: dict,
    neutral_site: bool = False,
) -> dict:
    """
    Generate CFB matchup features.

    IMPORTANT:
    If either team is missing SP+ data, SP-derived values
    are neutralized rather than manufacturing an advantage.
    """

    home = _get_cfb_sp_team(
        home_team,
        sp_lookup,
    )

    away = _get_cfb_sp_team(
        away_team,
        sp_lookup,
    )


    home_has_sp = (
        home is not None
    )

    away_has_sp = (
        away is not None
    )

    sp_data_complete = (
        home_has_sp
        and
        away_has_sp
    )


    hfa = (
        0.0
        if neutral_site
        else 3.0
    )


    # --------------------------------------------------------
    # Both teams have valid SP+
    # --------------------------------------------------------

    if sp_data_complete:

        home_overall = (
            home[
                "sp_overall"
            ]
        )

        away_overall = (
            away[
                "sp_overall"
            ]
        )

        home_offense = (
            home[
                "sp_offense"
            ]
        )

        home_defense = (
            home[
                "sp_defense"
            ]
        )

        away_offense = (
            away[
                "sp_offense"
            ]
        )

        away_defense = (
            away[
                "sp_defense"
            ]
        )


        sp_diff = (
            home_overall
            -
            away_overall
        )


        home_matchup = (
            home_offense
            -
            away_defense
        )

        away_matchup = (
            away_offense
            -
            home_defense
        )


    # --------------------------------------------------------
    # Missing SP+ for either team
    # --------------------------------------------------------

    else:

        # Neutralize SP features.
        # Do not pretend an unrated team has SP = 0.

        home_overall = 0.0
        away_overall = 0.0

        home_offense = 0.0
        home_defense = 0.0

        away_offense = 0.0
        away_defense = 0.0

        sp_diff = 0.0

        home_matchup = 0.0
        away_matchup = 0.0


    return {

        "sp_diff":
            sp_diff,

        "home_sp_overall":
            home_overall,

        "away_sp_overall":
            away_overall,

        "home_sp_offense":
            home_offense,

        "home_sp_defense":
            home_defense,

        "away_sp_offense":
            away_offense,

        "away_sp_defense":
            away_defense,

        "off_def_matchup_home":
            home_matchup,

        "off_def_matchup_away":
            away_matchup,

        "home_field_advantage":
            hfa,

        "predicted_home_off_contribution":
            (
                home_offense
                +
                hfa
                if sp_data_complete
                else hfa
            ),

        "predicted_away_off_contribution":
            (
                away_offense
                if sp_data_complete
                else 0.0
            ),

        "neutral_site":
            neutral_site,

        # Diagnostics / UI
        "home_has_sp_data":
            home_has_sp,

        "away_has_sp_data":
            away_has_sp,

        "sp_data_complete":
            sp_data_complete,

        "home_sp_team_name":
            (
                home.get(
                    "team"
                )
                if home
                else None
            ),

        "away_sp_team_name":
            (
                away.get(
                    "team"
                )
                if away
                else None
            ),
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

    Critical rule:
    Historical games are skipped when either team lacks
    SP+ data for that season.

    This avoids training the model with artificial
    zero/default SP+ ratings.
    """

    rows = []

    skipped_missing_score = 0
    skipped_missing_team = 0

    skipped_missing_sp = 0

    skipped_home_sp = 0
    skipped_away_sp = 0


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
            or
            away_points is None
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
            or
            not away_team
        ):

            skipped_missing_team += 1
            continue


        home_sp = _get_cfb_sp_team(
            home_team,
            sp_lookup,
        )

        away_sp = _get_cfb_sp_team(
            away_team,
            sp_lookup,
        )


        if home_sp is None:

            skipped_home_sp += 1


        if away_sp is None:

            skipped_away_sp += 1


        if (
            home_sp is None
            or
            away_sp is None
        ):

            skipped_missing_sp += 1
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


        # Extra safety.
        if not features.get(
            "sp_data_complete",
            False,
        ):

            skipped_missing_sp += 1
            continue


        features[
            "margin"
        ] = (
            home_points
            -
            away_points
        )

        features[
            "total"
        ] = (
            home_points
            +
            away_points
        )


        features[
            "home_score"
        ] = (
            home_points
        )

        features[
            "away_score"
        ] = (
            away_points
        )


        features[
            "season"
        ] = _value(
            game,
            "season",
            "season",
        )

        features[
            "week"
        ] = _value(
            game,
            "week",
            "week",
        )


        features[
            "home_team"
        ] = home_team

        features[
            "away_team"
        ] = away_team


        rows.append(
            features
        )


    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # Diagnostics
    # ========================================================

    df.attrs[
        "skipped_missing_score"
    ] = skipped_missing_score


    df.attrs[
        "skipped_missing_team"
    ] = skipped_missing_team


    df.attrs[
        "skipped_missing_sp"
    ] = skipped_missing_sp


    df.attrs[
        "skipped_home_sp"
    ] = skipped_home_sp


    df.attrs[
        "skipped_away_sp"
    ] = skipped_away_sp


    return df
