"""
card_engine.py

Generates the weekly Prime Picks card:
  1. Pull full NFL/CFB weekly schedule
  2. Run every game through the prediction model
  3. Fetch Vegas lines
  4. Apply roster adjustments
  5. Apply injury adjustments
  6. Apply line movement features
  7. Rank games by confidence-adjusted edge size
  8. Explain WHY Prime Picks likes a play

CFB prediction behavior:
- Both teams have SP+ -> trained CFB model
- Missing SP+ for either team -> opponent-adjusted SRS fallback

Confidence behavior:
- Trained model edges retain normal scoring/ranking
- Opponent-adjusted CFB fallback edges retain raw Vegas disagreement
- Fallback edges receive a reduced ranking score
- Fallback games cannot be presented as high-confidence Strong Edges

Explanation behavior:
- Position-group factors come directly from roster_engine ratings/weights
- Home-field value comes from model inputs
- Injury values come from injury_engine adjustments
- SP+/SRS values come from existing model features/diagnostics
- Vegas disagreement comes from calculated disparity
- Sharp/steam signals come from line movement snapshots
"""

import logging
import math

from typing import Optional
from datetime import datetime

from data_fetcher import (
    get_nfl_schedule_upcoming,
    get_cfb_upcoming,
)

from lines_fetcher import (
    get_lines,
    build_lines_lookup,
    find_line,
)

from feature_engine import (
    build_nfl_matchup_features,
    build_cfb_matchup_features,
)

from models import predictor
from roster_engine import roster_engine
from injury_engine import injury_engine
from line_snapshotter import snapshotter


logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _normal_cdf(value: float) -> float:
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


def _factor_impact(
    points: float,
) -> str:
    """
    Convert factor point magnitude into frontend impact tier.
    """

    magnitude = abs(
        float(points or 0)
    )

    if magnitude >= 2.0:
        return "high"

    if magnitude >= 0.75:
        return "medium"

    return "low"


def _safe_float(
    value,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# CFB opponent-adjusted fallback prediction
# ============================================================

def predict_cfb_fallback(
    home_team: str,
    away_team: str,
    team_stats: dict,
    neutral_site: bool = False,
) -> tuple[dict, dict]:

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

    raw_home_score = (
        (
            home.get(
                "avg_pts_for",
                27.0,
            )
            +
            away.get(
                "avg_pts_against",
                27.0,
            )
        )
        / 2.0
    )

    raw_away_score = (
        (
            away.get(
                "avg_pts_for",
                27.0,
            )
            +
            home.get(
                "avg_pts_against",
                27.0,
            )
        )
        / 2.0
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

    power_margin = (
        power_edge
        +
        hfa
    )

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

    predicted_total = _clip(
        raw_total,
        34.0,
        85.0,
    )

    predicted_margin = _clip(
        predicted_margin,
        -45.0,
        45.0,
    )

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

    margin_rmse = 16.0
    total_rmse = 19.0

    margin_lo = (
        predicted_margin
        -
        1.28 * margin_rmse
    )

    margin_hi = (
        predicted_margin
        +
        1.28 * margin_rmse
    )

    total_lo = (
        predicted_total
        -
        1.28 * total_rmse
    )

    total_hi = (
        predicted_total
        +
        1.28 * total_rmse
    )

    home_win_prob = (
        _normal_cdf(
            predicted_margin
            /
            margin_rmse
        )
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
            ) > 0,

        "away_history_available":
            away.get(
                "games",
                0,
            ) > 0,

        "fallback_method":
            "opponent_adjusted_srs",
    }

    return (
        prediction,
        diagnostics,
    )


# ============================================================
# Adjustments
# ============================================================

def apply_all_adjustments(
    features: dict,
    home_team: str,
    away_team: str,
    league: str,
) -> tuple[dict, dict]:

    adjusted = features.copy()

    summary = {
        "home_roster_adj": 0.0,
        "away_roster_adj": 0.0,

        "home_injury_adj": 0.0,
        "away_injury_adj": 0.0,

        "home_injuries": [],
        "away_injuries": [],

        "home_cascades": [],
        "away_cascades": [],

        "movement": {},
    }

    home_roster_adj = (
        roster_engine.get_team_adjustment(
            home_team
        )
        or 0.0
    )

    away_roster_adj = (
        roster_engine.get_team_adjustment(
            away_team
        )
        or 0.0
    )

    home_inj = (
        injury_engine.get_injury_adjustment(
            home_team,
            league,
            roster_engine,
        )
    )

    away_inj = (
        injury_engine.get_injury_adjustment(
            away_team,
            league,
            roster_engine,
        )
    )

    home_injury_adj = (
        home_inj.get(
            "adjustment",
            0.0,
        )
    )

    away_injury_adj = (
        away_inj.get(
            "adjustment",
            0.0,
        )
    )

    movement_feats = (
        snapshotter.get_model_features(
            home_team,
            away_team,
        )
    )

    movement_data = (
        snapshotter.get_movement(
            home_team,
            away_team,
        )
    )

    adjusted[
        "home_roster_adj"
    ] = home_roster_adj

    adjusted[
        "away_roster_adj"
    ] = away_roster_adj

    adjusted[
        "home_injury_adj"
    ] = home_injury_adj

    adjusted[
        "away_injury_adj"
    ] = away_injury_adj

    if "home_margin_avg" in adjusted:

        adjusted[
            "home_margin_avg"
        ] = (
            features.get(
                "home_margin_avg",
                0,
            )
            +
            home_roster_adj
            +
            home_injury_adj
        )

        adjusted[
            "away_margin_avg"
        ] = (
            features.get(
                "away_margin_avg",
                0,
            )
            +
            away_roster_adj
            +
            away_injury_adj
        )

    if (
        features.get(
            "sp_data_complete",
            False,
        )
        and
        "home_sp_overall"
        in adjusted
    ):

        adjusted[
            "home_sp_overall"
        ] = (
            features.get(
                "home_sp_overall",
                0,
            )
            +
            home_roster_adj
            +
            home_injury_adj
        )

        adjusted[
            "away_sp_overall"
        ] = (
            features.get(
                "away_sp_overall",
                0,
            )
            +
            away_roster_adj
            +
            away_injury_adj
        )

        adjusted[
            "sp_diff"
        ] = (
            adjusted[
                "home_sp_overall"
            ]
            -
            adjusted[
                "away_sp_overall"
            ]
        )

    adjusted.update(
        movement_feats
    )

    summary.update({
        "home_roster_adj":
            home_roster_adj,

        "away_roster_adj":
            away_roster_adj,

        "home_injury_adj":
            home_injury_adj,

        "away_injury_adj":
            away_injury_adj,

        "home_injuries":
            home_inj.get(
                "affected_players",
                [],
            ),

        "away_injuries":
            away_inj.get(
                "affected_players",
                [],
            ),

        "home_cascades":
            home_inj.get(
                "depth_chart_cascades",
                [],
            ),

        "away_cascades":
            away_inj.get(
                "depth_chart_cascades",
                [],
            ),

        "movement":
            movement_data,
    })

    return (
        adjusted,
        summary,
    )


# ============================================================
# Fallback roster / injury adjustment
# ============================================================

def apply_fallback_margin_adjustments(
    prediction: dict,
    adj_summary: dict,
) -> dict:

    result = prediction.copy()

    home_adj = (
        adj_summary.get(
            "home_roster_adj",
            0,
        )
        +
        adj_summary.get(
            "home_injury_adj",
            0,
        )
    )

    away_adj = (
        adj_summary.get(
            "away_roster_adj",
            0,
        )
        +
        adj_summary.get(
            "away_injury_adj",
            0,
        )
    )

    net_adjustment = (
        home_adj
        -
        away_adj
    )

    if abs(
        net_adjustment
    ) < 0.01:
        return result

    home_score = (
        result.get(
            "predicted_home_score",
            0,
        )
        +
        net_adjustment / 2.0
    )

    away_score = (
        result.get(
            "predicted_away_score",
            0,
        )
        -
        net_adjustment / 2.0
    )

    home_score = max(
        0.0,
        home_score,
    )

    away_score = max(
        0.0,
        away_score,
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

    result[
        "predicted_home_score"
    ] = round(
        home_score,
        1,
    )

    result[
        "predicted_away_score"
    ] = round(
        away_score,
        1,
    )

    result[
        "predicted_margin"
    ] = round(
        margin,
        1,
    )

    result[
        "predicted_total"
    ] = round(
        total,
        1,
    )

    margin_rmse = 16.0

    result[
        "home_win_prob"
    ] = round(
        _normal_cdf(
            margin
            /
            margin_rmse
        ),
        3,
    )

    result[
        "margin_80_lo"
    ] = round(
        margin
        -
        1.28
        *
        margin_rmse,
        1,
    )

    result[
        "margin_80_hi"
    ] = round(
        margin
        +
        1.28
        *
        margin_rmse,
        1,
    )

    return result


# ============================================================
# Vegas disparity / edge scoring
# ============================================================

def calculate_disparity(
    prediction: dict,
    line: Optional[dict],
    movement: dict,
) -> dict:

    if not line:

        return {
            "spread_disparity":
                None,

            "total_disparity":
                None,

            "edge_score":
                None,

            "spread_edge_type":
                None,

            "total_edge_type":
                None,

            "has_line":
                False,
        }

    our_margin = (
        prediction.get(
            "predicted_margin",
            0,
        )
    )

    our_total = (
        prediction.get(
            "predicted_total",
            0,
        )
    )

    vegas_spread = (
        line.get(
            "spread"
        )
    )

    vegas_total = (
        line.get(
            "total"
        )
    )

    result = {
        "has_line":
            True
    }

    if vegas_spread is not None:

        vegas_home_margin = (
            -float(
                vegas_spread
            )
        )

        spread_disp = (
            our_margin
            -
            vegas_home_margin
        )

        result[
            "spread_disparity"
        ] = round(
            spread_disp,
            1,
        )

        result[
            "vegas_spread"
        ] = vegas_spread

        result[
            "vegas_home_margin"
        ] = round(
            vegas_home_margin,
            1,
        )

        if abs(
            spread_disp
        ) < 1.5:

            result[
                "spread_edge_type"
            ] = "neutral"

        elif spread_disp > 0:

            result[
                "spread_edge_type"
            ] = "lean_home"

        else:

            result[
                "spread_edge_type"
            ] = "lean_away"

    else:

        result[
            "spread_disparity"
        ] = None

        result[
            "vegas_spread"
        ] = None

        result[
            "vegas_home_margin"
        ] = None

        result[
            "spread_edge_type"
        ] = None

    if vegas_total is not None:

        total_disp = (
            our_total
            -
            float(
                vegas_total
            )
        )

        result[
            "total_disparity"
        ] = round(
            total_disp,
            1,
        )

        result[
            "vegas_total"
        ] = vegas_total

        if abs(
            total_disp
        ) < 2.0:

            result[
                "total_edge_type"
            ] = "neutral"

        elif total_disp > 0:

            result[
                "total_edge_type"
            ] = "lean_over"

        else:

            result[
                "total_edge_type"
            ] = "lean_under"

    else:

        result[
            "total_disparity"
        ] = None

        result[
            "vegas_total"
        ] = None

        result[
            "total_edge_type"
        ] = None

    spread_score = (
        abs(
            result.get(
                "spread_disparity"
            )
            or 0
        )
        *
        3.0
    )

    total_score = (
        abs(
            result.get(
                "total_disparity"
            )
            or 0
        )
        *
        1.5
    )

    base_edge = (
        spread_score
        +
        total_score
    )

    if movement.get(
        "has_movement_data"
    ):

        sharp = (
            movement.get(
                "sharp_signal",
                0.0,
            )
            or 0.0
        )

    else:

        sharp = 0.0

    steam = bool(
        movement.get(
            "steam_move",
            False,
        )
    )

    move_dir = (
        movement.get(
            "move_direction"
        )
    )

    edge_type = (
        result.get(
            "spread_edge_type",
            "neutral",
        )
    )

    aligned = (
        (
            move_dir
            ==
            "toward_home"
            and
            edge_type
            ==
            "lean_home"
        )
        or
        (
            move_dir
            ==
            "toward_away"
            and
            edge_type
            ==
            "lean_away"
        )
    )

    sharp_bonus = (
        sharp * 8
        if aligned
        else 0.0
    )

    steam_bonus = (
        5.0
        if steam
        else 0.0
    )

    result[
        "edge_score"
    ] = round(
        min(
            100.0,
            base_edge
            +
            sharp_bonus
            +
            steam_bonus,
        ),
        1,
    )

    result[
        "sharp_aligned"
    ] = aligned

    result[
        "sharp_signal"
    ] = round(
        sharp,
        3,
    )

    result[
        "steam_move"
    ] = steam

    if result[
        "edge_score"
    ] >= 15:

        result[
            "edge_label"
        ] = "🔥 Strong Edge"

    elif result[
        "edge_score"
    ] >= 8:

        result[
            "edge_label"
        ] = "⚡ Moderate Edge"

    elif result[
        "edge_score"
    ] >= 3:

        result[
            "edge_label"
        ] = "→ Slight Lean"

    else:

        result[
            "edge_label"
        ] = "— Neutral"

    return result


# ============================================================
# Confidence weighting
# ============================================================

def apply_prediction_confidence(
    disparity: dict,
    prediction_mode: str,
    league: str,
) -> dict:

    result = disparity.copy()

    edge_score = (
        result.get(
            "edge_score"
        )
    )

    if edge_score is None:

        result[
            "raw_edge_score"
        ] = None

        result[
            "ranking_score"
        ] = None

        result[
            "confidence"
        ] = "unrated"

        result[
            "low_confidence"
        ] = False

        return result

    result[
        "raw_edge_score"
    ] = edge_score

    result[
        "ranking_score"
    ] = edge_score

    result[
        "low_confidence"
    ] = False

    if (
        league.upper()
        ==
        "CFB"
        and
        prediction_mode
        ==
        "opponent_adjusted_fallback"
    ):

        result[
            "confidence"
        ] = "low"

        result[
            "low_confidence"
        ] = True

        result[
            "ranking_score"
        ] = round(
            edge_score
            *
            0.40,
            1,
        )

        if edge_score >= 15:

            result[
                "edge_label"
            ] = "⚠ SRS Fallback Edge"

        elif edge_score >= 8:

            result[
                "edge_label"
            ] = "⚠ SRS Fallback Lean"

        elif edge_score >= 3:

            result[
                "edge_label"
            ] = "→ SRS Slight Lean"

        else:

            result[
                "edge_label"
            ] = "— Neutral"

        return result

    if (
        league.upper()
        ==
        "CFB"
        and
        prediction_mode
        ==
        "historical_fallback"
    ):

        result[
            "confidence"
        ] = "low"

        result[
            "low_confidence"
        ] = True

        result[
            "ranking_score"
        ] = round(
            edge_score
            *
            0.25,
            1,
        )

        result[
            "edge_label"
        ] = "⚠ Legacy Fallback Lean"

        return result

    if prediction_mode in (
        "trained_model",
        "trained_sp_model",
    ):

        result[
            "confidence"
        ] = "standard"

    else:

        result[
            "confidence"
        ] = "low"

        result[
            "low_confidence"
        ] = True

    return result


# ============================================================
# Injury notes
# ============================================================

def _format_injury_notes(
    injuries: list,
    cascades: list,
    team: str,
) -> list[str]:

    notes = []

    for injury in injuries:

        status = injury.get(
            "status",
            "",
        )

        name = injury.get(
            "name",
            "Unknown player",
        )

        position = injury.get(
            "position_group",
            "",
        )

        if status in (
            "out",
            "ir",
            "pup",
            "suspended",
        ):

            notes.append(
                f"❌ {name} "
                f"({position}) — "
                f"{status.upper()}"
            )

        elif status == "doubtful":

            notes.append(
                f"⚠ {name} "
                f"({position}) — Doubtful"
            )

        elif status == "questionable":

            notes.append(
                f"? {name} "
                f"({position}) — Questionable"
            )

    for cascade in cascades:

        starter_out = cascade.get(
            "starter_out",
            "Starter",
        )

        backup_in = cascade.get(
            "backup_in",
            "Backup",
        )

        position = cascade.get(
            "position_group",
            "",
        )

        notes.append(
            f"↓ {starter_out} out → "
            f"{backup_in} in "
            f"({position})"
        )

    return notes


# ============================================================
# NEW: Play explanation builder
# ============================================================

def build_play_explanation(
    home_team: str,
    away_team: str,
    league: str,
    neutral_site: bool,
    features: dict,
    prediction: dict,
    disparity: dict,
    adj_summary: dict,
    prediction_mode: str,
    fallback_diagnostics: Optional[dict] = None,
) -> dict:
    """
    Build a human-readable model explanation using actual
    model/roster/injury/market inputs.

    No AI-generated or fabricated point values are used here.
    """

    factors = []

    # --------------------------------------------------------
    # 1. POSITION GROUP FACTORS
    # --------------------------------------------------------

    try:

        roster_factors = (
            roster_engine
            .get_matchup_explanation_factors(
                home_team,
                away_team,
                minimum_points=0.05,
                limit=5,
            )
        )

        for factor in roster_factors:

            factors.append({
                "label":
                    factor.get(
                        "label",
                        "Roster Matchup",
                    ),

                "team":
                    factor.get(
                        "team"
                    ),

                "points":
                    round(
                        _safe_float(
                            factor.get(
                                "points",
                                0,
                            )
                        ),
                        2,
                    ),

                "signed_points":
                    round(
                        _safe_float(
                            factor.get(
                                "signed_points",
                                0,
                            )
                        ),
                        2,
                    ),

                "impact":
                    factor.get(
                        "impact",
                        "low",
                    ),

                "detail":
                    factor.get(
                        "detail",
                        "",
                    ),

                "source":
                    "roster",

                "group":
                    factor.get(
                        "group"
                    ),

                "top_players":
                    factor.get(
                        "top_players",
                        [],
                    ),
            })

    except Exception as exc:

        logger.debug(
            "Could not build position explanation for %s @ %s: %s",
            away_team,
            home_team,
            exc,
        )

    # --------------------------------------------------------
    # 2. INJURY FACTORS
    # --------------------------------------------------------

    home_injury_adj = _safe_float(
        adj_summary.get(
            "home_injury_adj",
            0,
        )
    )

    away_injury_adj = _safe_float(
        adj_summary.get(
            "away_injury_adj",
            0,
        )
    )

    injury_edge = (
        home_injury_adj
        -
        away_injury_adj
    )

    if abs(
        injury_edge
    ) >= 0.10:

        favored_team = (
            home_team
            if injury_edge > 0
            else away_team
        )

        disadvantaged_team = (
            away_team
            if injury_edge > 0
            else home_team
        )

        magnitude = abs(
            injury_edge
        )

        factors.append({
            "label":
                "Injury Advantage",

            "team":
                favored_team,

            "points":
                round(
                    magnitude,
                    2,
                ),

            "signed_points":
                round(
                    injury_edge,
                    2,
                ),

            "impact":
                _factor_impact(
                    magnitude
                ),

            "detail":
                (
                    f"{favored_team} gains approximately "
                    f"{magnitude:.2f} model points relative to "
                    f"{disadvantaged_team} from current injury adjustments."
                ),

            "source":
                "injuries",
        })

    # --------------------------------------------------------
    # 3. HOME FIELD
    # --------------------------------------------------------

    if not neutral_site:

        hfa = 3.0

        # Prefer actual value if the feature set exposes it.
        if "home_field" in features:

            feature_hfa = _safe_float(
                features.get(
                    "home_field",
                    3.0,
                ),
                3.0,
            )

            if abs(
                feature_hfa
            ) > 0:
                hfa = feature_hfa

        elif fallback_diagnostics:

            hfa = _safe_float(
                fallback_diagnostics.get(
                    "home_field_advantage",
                    3.0,
                ),
                3.0,
            )

        factors.append({
            "label":
                "Home Field",

            "team":
                home_team,

            "points":
                round(
                    abs(
                        hfa
                    ),
                    2,
                ),

            "signed_points":
                round(
                    hfa,
                    2,
                ),

            "impact":
                _factor_impact(
                    hfa
                ),

            "detail":
                (
                    f"{home_team} receives a "
                    f"{hfa:+.1f} point home-field adjustment."
                ),

            "source":
                "model",
        })

    # --------------------------------------------------------
    # 4. CFB SP+ GAP
    # --------------------------------------------------------

    if (
        league.upper() == "CFB"
        and
        prediction_mode == "trained_sp_model"
    ):

        home_sp = features.get(
            "home_sp_overall"
        )

        away_sp = features.get(
            "away_sp_overall"
        )

        if (
            home_sp is not None
            and
            away_sp is not None
        ):

            sp_gap = (
                _safe_float(
                    home_sp
                )
                -
                _safe_float(
                    away_sp
                )
            )

            if abs(
                sp_gap
            ) >= 0.5:

                favored_team = (
                    home_team
                    if sp_gap > 0
                    else away_team
                )

                factors.append({
                    "label":
                        "SP+ Rating Gap",

                    "team":
                        favored_team,

                    "points":
                        round(
                            abs(
                                sp_gap
                            ),
                            2,
                        ),

                    "signed_points":
                        round(
                            sp_gap,
                            2,
                        ),

                    "impact":
                        _factor_impact(
                            sp_gap
                        ),

                    "detail":
                        (
                            f"{favored_team} has the stronger SP+ profile "
                            f"by {abs(sp_gap):.1f} rating points."
                        ),

                    "source":
                        "sp_plus",
                })

    # --------------------------------------------------------
    # 5. CFB SRS / POWER GAP
    # --------------------------------------------------------

    if (
        league.upper() == "CFB"
        and
        prediction_mode
        ==
        "opponent_adjusted_fallback"
        and
        fallback_diagnostics
    ):

        power_edge = _safe_float(
            fallback_diagnostics.get(
                "power_rating_edge",
                0,
            )
        )

        if abs(
            power_edge
        ) >= 0.5:

            favored_team = (
                home_team
                if power_edge > 0
                else away_team
            )

            factors.append({
                "label":
                    "Opponent-Adjusted Power",

                "team":
                    favored_team,

                "points":
                    round(
                        abs(
                            power_edge
                        ),
                        2,
                    ),

                "signed_points":
                    round(
                        power_edge,
                        2,
                    ),

                "impact":
                    _factor_impact(
                        power_edge
                    ),

                "detail":
                    (
                        f"{favored_team} leads the opponent-adjusted "
                        f"SRS/power comparison by "
                        f"{abs(power_edge):.1f} rating points."
                    ),

                "source":
                    "srs",
            })

        raw_scoring_margin = _safe_float(
            fallback_diagnostics.get(
                "raw_scoring_margin",
                0,
            )
        )

        if abs(
            raw_scoring_margin
        ) >= 1.0:

            favored_team = (
                home_team
                if raw_scoring_margin > 0
                else away_team
            )

            factors.append({
                "label":
                    "Recent Scoring",

                "team":
                    favored_team,

                "points":
                    round(
                        abs(
                            raw_scoring_margin
                        ),
                        2,
                    ),

                "signed_points":
                    round(
                        raw_scoring_margin,
                        2,
                    ),

                "impact":
                    _factor_impact(
                        raw_scoring_margin
                    ),

                "detail":
                    (
                        f"Recent offensive/defensive scoring profiles "
                        f"favor {favored_team} by approximately "
                        f"{abs(raw_scoring_margin):.1f} points."
                    ),

                "source":
                    "recent_form",
            })

    # --------------------------------------------------------
    # 6. VEGAS SPREAD DISAGREEMENT
    # --------------------------------------------------------

    spread_disparity = (
        disparity.get(
            "spread_disparity"
        )
    )

    if spread_disparity is not None:

        spread_disparity = _safe_float(
            spread_disparity
        )

        if abs(
            spread_disparity
        ) >= 1.5:

            favored_team = (
                home_team
                if spread_disparity > 0
                else away_team
            )

            factors.append({
                "label":
                    "Model vs Vegas",

                "team":
                    favored_team,

                "points":
                    round(
                        abs(
                            spread_disparity
                        ),
                        2,
                    ),

                "signed_points":
                    round(
                        spread_disparity,
                        2,
                    ),

                "impact":
                    _factor_impact(
                        spread_disparity
                    ),

                "detail":
                    (
                        f"Prime Picks differs from the Vegas spread "
                        f"by {abs(spread_disparity):.1f} points "
                        f"in favor of {favored_team}."
                    ),

                "source":
                    "market",
            })

    # --------------------------------------------------------
    # 7. SHARP / STEAM MOVEMENT
    # --------------------------------------------------------

    movement = (
        adj_summary.get(
            "movement",
            {},
        )
        or {}
    )

    sharp_signal = _safe_float(
        movement.get(
            "sharp_signal",
            0,
        )
    )

    steam_move = bool(
        movement.get(
            "steam_move",
            False,
        )
    )

    move_direction = movement.get(
        "move_direction"
    )

    if (
        sharp_signal >= 0.40
        or
        steam_move
    ):

        movement_team = None

        if (
            move_direction
            ==
            "toward_home"
        ):
            movement_team = home_team

        elif (
            move_direction
            ==
            "toward_away"
        ):
            movement_team = away_team

        if movement_team:

            movement_strength = (
                sharp_signal * 2.0
            )

            if steam_move:
                movement_strength += 1.0

            factors.append({
                "label":
                    (
                        "Steam Move"
                        if steam_move
                        else
                        "Sharp Action"
                    ),

                "team":
                    movement_team,

                "points":
                    round(
                        movement_strength,
                        2,
                    ),

                "signed_points":
                    None,

                "impact":
                    (
                        "high"
                        if steam_move
                        else "medium"
                    ),

                "detail":
                    (
                        f"Recent betting-market movement is "
                        f"toward {movement_team}. "
                        f"Sharp signal: "
                        f"{sharp_signal * 100:.0f}%."
                    ),

                "source":
                    "market_movement",
            })

    # --------------------------------------------------------
    # SORT MOST MEANINGFUL FACTORS FIRST
    # --------------------------------------------------------

    factors.sort(
        key=lambda item:
            abs(
                _safe_float(
                    item.get(
                        "points",
                        0,
                    )
                )
            ),
        reverse=True,
    )

    # Keep the explanation readable.
    factors = factors[
        :8
    ]

    # --------------------------------------------------------
    # Determine recommended side
    # --------------------------------------------------------

    spread_edge_type = disparity.get(
        "spread_edge_type"
    )

    if (
        spread_edge_type
        ==
        "lean_home"
    ):

        recommended_team = (
            home_team
        )

    elif (
        spread_edge_type
        ==
        "lean_away"
    ):

        recommended_team = (
            away_team
        )

    else:

        predicted_margin = _safe_float(
            prediction.get(
                "predicted_margin",
                0,
            )
        )

        recommended_team = (
            home_team
            if predicted_margin >= 0
            else away_team
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    team_factor_count = sum(
        1
        for factor
        in factors
        if factor.get(
            "team"
        )
        ==
        recommended_team
    )

    if team_factor_count >= 3:

        summary = (
            f"Prime Picks favors {recommended_team} because "
            f"multiple model inputs align on the same side."
        )

    elif team_factor_count >= 1:

        summary = (
            f"Prime Picks sees an advantage for "
            f"{recommended_team} based on the matchup and market data."
        )

    else:

        summary = (
            "Prime Picks sees a market/model discrepancy, "
            "but the supporting factors are mixed."
        )

    return {
        "recommended_team":
            recommended_team,

        "summary":
            summary,

        "factor_count":
            len(
                factors
            ),

        "factors":
            factors,
    }


# ============================================================
# Weekly Card
# ============================================================

async def generate_weekly_card(
    league: str,
    week: int,
    season: Optional[int],
    nfl_team_stats: dict,
    cfb_sp_lookup: dict,
    cfb_team_stats: Optional[dict] = None,
) -> dict:

    league_upper = (
        league.upper()
    )

    cfb_team_stats = (
        cfb_team_stats
        or {}
    )

    logger.info(
        "Generating %s Week %s card...",
        league_upper,
        week,
    )

    # ========================================================
    # 1. Schedule
    # ========================================================

    if league_upper == "NFL":

        games = (
            await get_nfl_schedule_upcoming(
                week=week,
                season=season,
            )
        )

    else:

        games = (
            await get_cfb_upcoming(
                week=week,
                season=season,
            )
        )

    logger.info(
        "%s Week %s schedule returned %s raw games",
        league_upper,
        week,
        len(
            games
        ),
    )

    if not games:

        return {
            "league":
                league_upper,

            "week":
                week,

            "season":
                season,

            "generated_at":
                datetime.utcnow().isoformat(),

            "raw_schedule_games":
                0,

            "skipped_schedule_games":
                0,

            "total_games":
                0,

            "games_with_lines":
                0,

            "games_with_movement":
                0,

            "cfb_sp_model_games":
                0,

            "cfb_fallback_games":
                0,

            "snapshot_stats":
                snapshotter.get_snapshot_stats(),

            "games":
                [],

            "error":
                "No games found",
        }

    # ========================================================
    # 2. Vegas lines
    # ========================================================

    try:

        lines = (
            await get_lines(
                league=
                    league_upper,

                week=
                    week,
            )
        )

    except Exception as exc:

        logger.warning(
            "%s line fetch failed: %s",
            league_upper,
            exc,
        )

        lines = []

    logger.info(
        "%s lines returned: %s",
        league_upper,
        len(
            lines
        ),
    )

    lines_lookup = (
        build_lines_lookup(
            lines
        )
    )

    # ========================================================
    # 3. Fresh movement snapshot
    # ========================================================

    try:

        await snapshotter.take_snapshot(
            league_upper
        )

    except Exception as exc:

        logger.warning(
            "%s fresh snapshot failed: %s",
            league_upper,
            exc,
        )

    # ========================================================
    # 4. Build card
    # ========================================================

    card_games = []

    skipped_games = 0
    unmatched_lines = 0

    cfb_sp_model_games = 0
    cfb_fallback_games = 0

    for game in games:

        # ----------------------------------------------------
        # Normalize game fields
        # ----------------------------------------------------

        if league_upper == "NFL":

            home = (
                game.get(
                    "home_team"
                )
                or ""
            )

            away = (
                game.get(
                    "away_team"
                )
                or ""
            )

            neutral = bool(
                game.get(
                    "neutral_site",
                    False,
                )
            )

            game_date = (
                game.get(
                    "date"
                )
                or ""
            )

            venue = (
                game.get(
                    "venue"
                )
                or ""
            )

            game_id = (
                game.get(
                    "game_id"
                )
                or ""
            )

        else:

            home = (
                game.get(
                    "homeTeam"
                )
                or
                game.get(
                    "home_team"
                )
                or ""
            )

            away = (
                game.get(
                    "awayTeam"
                )
                or
                game.get(
                    "away_team"
                )
                or ""
            )

            neutral = bool(
                game.get(
                    "neutralSite",
                    game.get(
                        "neutral_site",
                        False,
                    ),
                )
            )

            game_date = (
                game.get(
                    "startDate"
                )
                or
                game.get(
                    "start_date"
                )
                or
                game.get(
                    "date"
                )
                or ""
            )

            venue = (
                game.get(
                    "venue"
                )
                or ""
            )

            game_id = (
                game.get(
                    "id"
                )
                or
                game.get(
                    "game_id"
                )
                or ""
            )

        if not home or not away:

            skipped_games += 1

            logger.warning(
                "%s game skipped — missing teams. game_id=%s",
                league_upper,
                game_id,
            )

            continue


        # ====================================================
        # EARLY LINE FILTER
        # Only process games that currently have Vegas lines.
        # This prevents CFB from running the full prediction /
        # roster / injury / explanation pipeline on hundreds
        # of irrelevant schedule records.
        # ====================================================

        line = find_line(
            lines_lookup,
            home,
            away,
        )

        if line is None:

            unmatched_lines += 1

            logger.debug(
                "%s game skipped — no Vegas line: %s @ %s",
                league_upper,
                away,
                home,
            )

            continue

        # ====================================================
        # 5. Feature generation
        # ====================================================

        if league_upper == "NFL":

            features = (
                build_nfl_matchup_features(
                    home,
                    away,
                    nfl_team_stats,
                    neutral_site=
                        neutral,
                )
            )

        else:

            features = (
                build_cfb_matchup_features(
                    home,
                    away,
                    cfb_sp_lookup,
                    neutral_site=
                        neutral,
                )
            )

        # ====================================================
        # 6. Adjustments
        # ====================================================

        (
            features,
            adj_summary,
        ) = apply_all_adjustments(
            features,
            home,
            away,
            league_upper,
        )

        # ====================================================
        # 7. Prediction
        # ====================================================

        fallback_diagnostics = None

        if league_upper == "NFL":

            prediction = (
                predictor.predict_nfl(
                    features
                )
            )

            prediction_mode = (
                "trained_model"
                if prediction.get(
                    "model_trained"
                )
                else
                "baseline"
            )

        else:

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

                cfb_sp_model_games += 1

            else:

                (
                    prediction,
                    fallback_diagnostics,
                ) = predict_cfb_fallback(
                    home,
                    away,
                    cfb_team_stats,
                    neutral_site=
                        neutral,
                )

                prediction = (
                    apply_fallback_margin_adjustments(
                        prediction,
                        adj_summary,
                    )
                )

                prediction_mode = (
                    "opponent_adjusted_fallback"
                )

                cfb_fallback_games += 1

        # ====================================================
        # 8. Vegas
        # ====================================================

       

        movement = (
            adj_summary[
                "movement"
            ]
        )

        disparity = (
            calculate_disparity(
                prediction,
                line,
                movement,
            )
        )

        disparity = (
            apply_prediction_confidence(
                disparity,
                prediction_mode,
                league_upper,
            )
        )

        # ====================================================
        # 9. Injuries
        # ====================================================

        home_injury_notes = (
            _format_injury_notes(
                adj_summary[
                    "home_injuries"
                ],
                adj_summary[
                    "home_cascades"
                ],
                home,
            )
        )

        away_injury_notes = (
            _format_injury_notes(
                adj_summary[
                    "away_injuries"
                ],
                adj_summary[
                    "away_cascades"
                ],
                away,
            )
        )

        # ====================================================
        # 10. Roster notes
        # ====================================================

        roster_notes = []

        if abs(
            adj_summary[
                "home_roster_adj"
            ]
        ) >= 0.5:

            direction = (
                "↑"
                if adj_summary[
                    "home_roster_adj"
                ] > 0
                else
                "↓"
            )

            roster_notes.append(
                f"{direction} "
                f"{home} roster adj: "
                f"{adj_summary['home_roster_adj']:+.1f} pts"
            )

        if abs(
            adj_summary[
                "away_roster_adj"
            ]
        ) >= 0.5:

            direction = (
                "↑"
                if adj_summary[
                    "away_roster_adj"
                ] > 0
                else
                "↓"
            )

            roster_notes.append(
                f"{direction} "
                f"{away} roster adj: "
                f"{adj_summary['away_roster_adj']:+.1f} pts"
            )

        # ====================================================
        # 11. Movement display
        # ====================================================

        mv_display = None

        if (
            movement.get(
                "has_movement_data"
            )
            and
            movement.get(
                "spread_move"
            )
            is not None
        ):

            mv_display = {
                "spread_open":
                    movement.get(
                        "spread_open"
                    ),

                "spread_current":
                    movement.get(
                        "spread_current"
                    ),

                "spread_move":
                    movement.get(
                        "spread_move"
                    ),

                "total_open":
                    movement.get(
                        "total_open"
                    ),

                "total_current":
                    movement.get(
                        "total_current"
                    ),

                "total_move":
                    movement.get(
                        "total_move"
                    ),

                "sharp_signal":
                    movement.get(
                        "sharp_signal",
                        0,
                    ),

                "steam_move":
                    movement.get(
                        "steam_move",
                        False,
                    ),

                "move_direction":
                    movement.get(
                        "move_direction"
                    ),

                "hours_tracked":
                    movement.get(
                        "hours_elapsed",
                        0,
                    ),
            }

        # ====================================================
        # 12. NEW: WHY PRIME PICKS LIKES THIS PLAY
        # ====================================================

        play_explanation = (
            build_play_explanation(
                home_team=
                    home,

                away_team=
                    away,

                league=
                    league_upper,

                neutral_site=
                    neutral,

                features=
                    features,

                prediction=
                    prediction,

                disparity=
                    disparity,

                adj_summary=
                    adj_summary,

                prediction_mode=
                    prediction_mode,

                fallback_diagnostics=
                    fallback_diagnostics,
            )
        )

        # ====================================================
        # 13. Response
        # ====================================================

        card_game = {
            "game_id":
                game_id,

            "home_team":
                home,

            "away_team":
                away,

            "date":
                game_date,

            "venue":
                venue,

            "neutral_site":
                neutral,

            "prediction_mode":
                prediction_mode,

            "confidence":
                disparity.get(
                    "confidence",
                    "unrated",
                ),

            "prediction":
                prediction,

            "disparity":
                disparity,

            "line_movement":
                mv_display,

            "home_injury_notes":
                home_injury_notes,

            "away_injury_notes":
                away_injury_notes,

            "roster_notes":
                roster_notes,

            "adjustments": {
                "home_roster":
                    adj_summary[
                        "home_roster_adj"
                    ],

                "away_roster":
                    adj_summary[
                        "away_roster_adj"
                    ],

                "home_injury":
                    adj_summary[
                        "home_injury_adj"
                    ],

                "away_injury":
                    adj_summary[
                        "away_injury_adj"
                    ],
            },

            # NEW
            "key_factors":
                play_explanation.get(
                    "factors",
                    [],
                ),

            # NEW
            "play_explanation":
                play_explanation,

            "league":
                league_upper,

            "week":
                week,

            "season":
                season,
        }

        if fallback_diagnostics is not None:

            card_game[
                "fallback_diagnostics"
            ] = fallback_diagnostics

        if (
            league_upper == "CFB"
            and
            not features.get(
                "sp_data_complete",
                False,
            )
        ):

            missing_sp_teams = []

            if not features.get(
                "home_has_sp_data",
                False,
            ):

                missing_sp_teams.append(
                    home
                )

            if not features.get(
                "away_has_sp_data",
                False,
            ):

                missing_sp_teams.append(
                    away
                )

            if missing_sp_teams:

                card_game[
                    "model_note"
                ] = (
                    "SP+ unavailable for "
                    +
                    ", ".join(
                        missing_sp_teams
                    )
                    +
                    "; opponent-adjusted SRS fallback used. "
                    "Edge confidence reduced."
                )

            else:

                card_game[
                    "model_note"
                ] = (
                    "Incomplete SP+ data; "
                    "opponent-adjusted SRS fallback used. "
                    "Edge confidence reduced."
                )

        card_games.append(
            card_game
        )

    # ========================================================
    # 14. Ranking
    # ========================================================

    card_games.sort(
        key=lambda game:
            (
                game[
                    "disparity"
                ].get(
                    "ranking_score"
                )
                or 0
            ),
        reverse=True,
    )

    # ========================================================
    # 15. Coverage
    # ========================================================

    lines_coverage = sum(
        1
        for game in card_games
        if game[
            "disparity"
        ][
            "has_line"
        ]
    )

    games_with_movement = sum(
        1
        for game in card_games
        if game[
            "line_movement"
        ]
        is not None
    )

    low_confidence_games = sum(
        1
        for game in card_games
        if game.get(
            "confidence"
        )
        ==
        "low"
    )

    standard_confidence_games = sum(
        1
        for game in card_games
        if game.get(
            "confidence"
        )
        ==
        "standard"
    )

    logger.info(
        (
            "%s Week %s card complete: "
            "%s raw, "
            "%s card games, "
            "%s lines, "
            "%s movement, "
            "%s skipped, "
            "%s unmatched lines, "
            "%s SP+ model, "
            "%s opponent-adjusted fallback, "
            "%s standard confidence, "
            "%s low confidence"
        ),
        league_upper,
        week,
        len(
            games
        ),
        len(
            card_games
        ),
        lines_coverage,
        games_with_movement,
        skipped_games,
        unmatched_lines,
        cfb_sp_model_games,
        cfb_fallback_games,
        standard_confidence_games,
        low_confidence_games,
    )

    return {
        "league":
            league_upper,

        "week":
            week,

        "season":
            season,

        "generated_at":
            datetime.utcnow().isoformat(),

        "raw_schedule_games":
            len(
                games
            ),

        "skipped_schedule_games":
            skipped_games,

        "unmatched_lines":
            unmatched_lines,

        "total_games":
            len(
                card_games
            ),

        "games_with_lines":
            lines_coverage,

        "games_with_movement":
            games_with_movement,

        "standard_confidence_games":
            standard_confidence_games,

        "low_confidence_games":
            low_confidence_games,

        "cfb_sp_model_games":
            (
                cfb_sp_model_games
                if league_upper
                ==
                "CFB"
                else 0
            ),

        "cfb_fallback_games":
            (
                cfb_fallback_games
                if league_upper
                ==
                "CFB"
                else 0
            ),

        "cfb_fallback_method":
            (
                "opponent_adjusted_srs"
                if league_upper
                ==
                "CFB"
                else None
            ),

        "snapshot_stats":
            snapshotter.get_snapshot_stats(),

        "games":
            card_games,
    }
