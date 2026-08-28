"""
card_engine.py

Generates the weekly Prime Picks card:
  1. Pull full NFL/CFB weekly schedule
  2. Run every game through the prediction model
  3. Fetch Vegas lines
  4. Apply roster adjustments (player moves)
  5. Apply injury adjustments (depth chart cascade)
  6. Apply line movement features (sharp signal)
  7. Rank games by edge size — largest gaps surface first

Disparity scoring:
  spread_disparity: our_margin vs vegas_spread
  total_disparity: our_total vs vegas_total
  edge_score: weighted sum
  sharp_signal: line movement intensity (0-1, higher = more sharp action)
"""

import logging
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
# Adjustments
# ============================================================

def apply_all_adjustments(
    features: dict,
    home_team: str,
    away_team: str,
    league: str,
) -> tuple[dict, dict]:
    """
    Apply roster, injury, and line movement adjustments
    to base model features.

    Returns:
        adjusted_features
        adjustment_summary
    """

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


    # --------------------------------------------------------
    # Roster adjustments
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Injury adjustments
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Line movement
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Apply adjustments
    # --------------------------------------------------------

    adjusted["home_roster_adj"] = (
        home_roster_adj
    )

    adjusted["away_roster_adj"] = (
        away_roster_adj
    )

    adjusted["home_injury_adj"] = (
        home_injury_adj
    )

    adjusted["away_injury_adj"] = (
        away_injury_adj
    )


    # --------------------------------------------------------
    # NFL readable margin adjustments
    # --------------------------------------------------------

    if "home_margin_avg" in adjusted:

        adjusted["home_margin_avg"] = (
            features.get(
                "home_margin_avg",
                0,
            )
            + home_roster_adj
            + home_injury_adj
        )

        adjusted["away_margin_avg"] = (
            features.get(
                "away_margin_avg",
                0,
            )
            + away_roster_adj
            + away_injury_adj
        )


    # --------------------------------------------------------
    # CFB readable SP+ adjustments
    # --------------------------------------------------------

    if "home_sp_overall" in adjusted:

        adjusted["home_sp_overall"] = (
            features.get(
                "home_sp_overall",
                0,
            )
            + home_roster_adj
            + home_injury_adj
        )

        adjusted["away_sp_overall"] = (
            features.get(
                "away_sp_overall",
                0,
            )
            + away_roster_adj
            + away_injury_adj
        )

        adjusted["sp_diff"] = (
            adjusted["home_sp_overall"]
            -
            adjusted["away_sp_overall"]
        )


    # --------------------------------------------------------
    # Sharp-money / movement model features
    # --------------------------------------------------------

    adjusted.update(
        movement_feats
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

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

    return adjusted, summary


# ============================================================
# Disparity / edge scoring
# ============================================================

def calculate_disparity(
    prediction: dict,
    line: Optional[dict],
    movement: dict,
) -> dict:
    """
    Compare Prime Picks prediction to Vegas line.

    Also incorporates line movement / sharp-money signal.
    """

    if not line:

        return {
            "spread_disparity": None,
            "total_disparity": None,
            "edge_score": None,

            "spread_edge_type": None,
            "total_edge_type": None,

            "has_line": False,
        }


    our_margin = prediction.get(
        "predicted_margin",
        0,
    )

    our_total = prediction.get(
        "predicted_total",
        0,
    )

    vegas_spread = line.get(
        "spread"
    )

    vegas_total = line.get(
        "total"
    )

    result = {
        "has_line": True
    }


    # --------------------------------------------------------
    # Spread disparity
    # --------------------------------------------------------

    if vegas_spread is not None:

        spread_disp = (
            our_margin
            -
            (-vegas_spread)
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


        if abs(
            spread_disp
        ) < 1.5:

            result[
                "spread_edge_type"
            ] = "neutral"


        elif (
            our_margin > 0
            and vegas_spread > 0
        ):

            result[
                "spread_edge_type"
            ] = "fade_away"


        elif (
            our_margin < 0
            and vegas_spread < 0
        ):

            result[
                "spread_edge_type"
            ] = "fade_home"


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
            "spread_edge_type"
        ] = None


    # --------------------------------------------------------
    # Total disparity
    # --------------------------------------------------------

    if vegas_total is not None:

        total_disp = (
            our_total
            -
            vegas_total
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
        ) < 2:

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


    # --------------------------------------------------------
    # Edge score base
    # --------------------------------------------------------

    spread_score = (
        abs(
            result.get(
                "spread_disparity"
            )
            or 0
        )
        * 3
    )

    total_score = (
        abs(
            result.get(
                "total_disparity"
            )
            or 0
        )
        * 1.5
    )

    base_edge = (
        spread_score
        +
        total_score
    )


    # --------------------------------------------------------
    # Sharp signal
    # --------------------------------------------------------

    if movement.get(
        "has_movement_data"
    ):

        sharp = movement.get(
            "sharp_signal",
            0.0,
        )

    else:

        sharp = 0.0


    steam = movement.get(
        "steam_move",
        False,
    )

    move_dir = movement.get(
        "move_direction"
    )

    edge_type = result.get(
        "spread_edge_type",
        "neutral",
    )


    aligned = (

        (
            move_dir
            ==
            "toward_home"
        )

        and

        edge_type in (
            "lean_home",
            "fade_away",
        )

    ) or (

        (
            move_dir
            ==
            "toward_away"
        )

        and

        edge_type in (
            "lean_away",
            "fade_home",
        )

    )


    sharp_bonus = (
        sharp * 8
        if aligned
        else 0
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
            100,
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


    # --------------------------------------------------------
    # Human-readable edge label
    # --------------------------------------------------------

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
# Weekly Card
# ============================================================

async def generate_weekly_card(
    league: str,
    week: int,
    season: Optional[int],
    nfl_team_stats: dict,
    cfb_sp_lookup: dict,
) -> dict:
    """
    Generate full weekly Prime Picks slate.

    Supports:
    - ESPN-normalized NFL schedule fields
    - CollegeFootballData camelCase schedule fields
    - legacy snake_case CFB fields
    """

    league_upper = (
        league.upper()
    )

    logger.info(
        "Generating %s Week %s card...",
        league_upper,
        week,
    )


    # ========================================================
    # 1. Fetch schedule
    # ========================================================

    if league_upper == "NFL":

        games = await get_nfl_schedule_upcoming(
            week=week,
            season=season,
        )

    else:

        games = await get_cfb_upcoming(
            week=week,
            season=season,
        )


    logger.info(
        "%s Week %s schedule returned %s raw games",
        league_upper,
        week,
        len(games),
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

            "snapshot_stats":
                snapshotter.get_snapshot_stats(),

            "games":
                [],

            "error":
                "No games found",
        }


    # ========================================================
    # 2. Fetch current Vegas lines
    # ========================================================

    try:

        lines = await get_lines(
            league=league_upper,
            week=week,
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
        len(lines),
    )


    lines_lookup = (
        build_lines_lookup(
            lines
        )
    )


    # ========================================================
    # 3. Fresh snapshot
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


    for game in games:

        # ----------------------------------------------------
        # Normalize schedule field names
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

            # Current CollegeFootballData API format.
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


        # ----------------------------------------------------
        # Skip unusable games
        # ----------------------------------------------------

        if not home or not away:

            skipped_games += 1

            logger.warning(
                "%s game skipped — missing teams. "
                "game_id=%s keys=%s",
                league_upper,
                game_id,
                list(
                    game.keys()
                )[:20],
            )

            continue


        # ====================================================
        # 5. Base features
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
        # 6. Roster + injuries + movement
        # ====================================================

        features, adj_summary = (
            apply_all_adjustments(
                features,
                home,
                away,
                league_upper,
            )
        )


        # ====================================================
        # 7. Prediction
        # ====================================================

        if league_upper == "NFL":

            prediction = (
                predictor.predict_nfl(
                    features
                )
            )


        else:

            prediction = (
                predictor.predict_cfb(
                    features
                )
            )


        # ====================================================
        # 8. Match Vegas line
        # ====================================================

        line = find_line(
            lines_lookup,
            home,
            away,
        )


        if line is None:

            unmatched_lines += 1

            logger.debug(
                "%s line not matched: %s @ %s",
                league_upper,
                away,
                home,
            )


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


        # ====================================================
        # 9. Injury notes
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
                else "↓"
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
                else "↓"
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
        # 12. Add card game
        # ====================================================

        card_games.append({
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

            "league":
                league_upper,

            "week":
                week,

            "season":
                season,
        })


    # ========================================================
    # 13. Rank by edge
    # ========================================================

    card_games.sort(

        key=lambda game:
            game[
                "disparity"
            ].get(
                "edge_score"
            )
            or 0,

        reverse=True,
    )


    # ========================================================
    # 14. Coverage
    # ========================================================

    lines_coverage = sum(

        1

        for game
        in card_games

        if game[
            "disparity"
        ][
            "has_line"
        ]
    )


    games_with_movement = sum(

        1

        for game
        in card_games

        if game[
            "line_movement"
        ]
        is not None
    )


    logger.info(
        "%s Week %s card complete: "
        "%s raw, "
        "%s card games, "
        "%s lines, "
        "%s movement, "
        "%s skipped, "
        "%s unmatched lines",
        league_upper,
        week,
        len(games),
        len(card_games),
        lines_coverage,
        games_with_movement,
        skipped_games,
        unmatched_lines,
    )


    # ========================================================
    # Final response
    # ========================================================

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
            len(games),

        "skipped_schedule_games":
            skipped_games,

        "unmatched_lines":
            unmatched_lines,

        "total_games":
            len(card_games),

        "games_with_lines":
            lines_coverage,

        "games_with_movement":
            games_with_movement,

        "snapshot_stats":
            snapshotter.get_snapshot_stats(),

        "games":
            card_games,
    }
