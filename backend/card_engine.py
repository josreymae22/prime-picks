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

import asyncio
import logging
from typing import Optional
from datetime import datetime

from data_fetcher import get_nfl_schedule_upcoming, get_cfb_upcoming
from lines_fetcher import get_lines, build_lines_lookup, find_line
from feature_engine import build_nfl_matchup_features, build_cfb_matchup_features
from models import predictor
from roster_engine import roster_engine
from injury_engine import injury_engine
from line_snapshotter import snapshotter

logger = logging.getLogger(__name__)


def apply_all_adjustments(
    features: dict,
    home_team: str,
    away_team: str,
    league: str,
) -> tuple[dict, dict]:
    """
    Apply roster, injury, and line movement adjustments to base features.
    Returns (adjusted_features, adjustment_summary).
    """
    adjusted = features.copy()
    summary = {
        "home_roster_adj": 0.0, "away_roster_adj": 0.0,
        "home_injury_adj": 0.0, "away_injury_adj": 0.0,
        "home_injuries": [], "away_injuries": [],
        "home_cascades": [], "away_cascades": [],
        "movement": {},
    }

    # ── Roster adjustments ────────────────────
    home_roster_adj = roster_engine.get_team_adjustment(home_team) or 0.0
    away_roster_adj = roster_engine.get_team_adjustment(away_team) or 0.0

    # ── Injury adjustments ────────────────────
    home_inj = injury_engine.get_injury_adjustment(home_team, league, roster_engine)
    away_inj = injury_engine.get_injury_adjustment(away_team, league, roster_engine)
    home_injury_adj = home_inj["adjustment"]
    away_injury_adj = away_inj["adjustment"]

    # ── Line movement features ────────────────
    movement_feats = snapshotter.get_model_features(home_team, away_team)
    movement_data = snapshotter.get_movement(home_team, away_team)

    # Apply to features
    adjusted["home_roster_adj"] = home_roster_adj
    adjusted["away_roster_adj"] = away_roster_adj
    adjusted["home_injury_adj"] = home_injury_adj
    adjusted["away_injury_adj"] = away_injury_adj

    # Also adjust margin/SP averages for readability
    if "home_margin_avg" in adjusted:
        adjusted["home_margin_avg"] = (
            features.get("home_margin_avg", 0) + home_roster_adj + home_injury_adj
        )
        adjusted["away_margin_avg"] = (
            features.get("away_margin_avg", 0) + away_roster_adj + away_injury_adj
        )
    if "home_sp_overall" in adjusted:
        adjusted["home_sp_overall"] = features.get("home_sp_overall", 0) + home_roster_adj + home_injury_adj
        adjusted["away_sp_overall"] = features.get("away_sp_overall", 0) + away_roster_adj + away_injury_adj
        adjusted["sp_diff"] = adjusted["home_sp_overall"] - adjusted["away_sp_overall"]

    # Sharp money features
    adjusted.update(movement_feats)

    summary.update({
        "home_roster_adj": home_roster_adj,
        "away_roster_adj": away_roster_adj,
        "home_injury_adj": home_injury_adj,
        "away_injury_adj": away_injury_adj,
        "home_injuries": home_inj.get("affected_players", []),
        "away_injuries": away_inj.get("affected_players", []),
        "home_cascades": home_inj.get("depth_chart_cascades", []),
        "away_cascades": away_inj.get("depth_chart_cascades", []),
        "movement": movement_data,
    })

    return adjusted, summary


def calculate_disparity(prediction: dict, line: Optional[dict], movement: dict) -> dict:
    """
    Compare our prediction to the Vegas line.
    Incorporates sharp signal into the edge scoring.
    """
    if not line:
        return {
            "spread_disparity": None, "total_disparity": None,
            "edge_score": None, "spread_edge_type": None,
            "total_edge_type": None, "has_line": False,
        }

    our_margin = prediction.get("predicted_margin", 0)
    our_total = prediction.get("predicted_total", 0)
    vegas_spread = line.get("spread")
    vegas_total = line.get("total")

    result = {"has_line": True}

    if vegas_spread is not None:
        spread_disp = our_margin - (-vegas_spread)
        result["spread_disparity"] = round(spread_disp, 1)
        result["vegas_spread"] = vegas_spread

        if abs(spread_disp) < 1.5:
            result["spread_edge_type"] = "neutral"
        elif our_margin > 0 and vegas_spread > 0:
            result["spread_edge_type"] = "fade_away"
        elif our_margin < 0 and vegas_spread < 0:
            result["spread_edge_type"] = "fade_home"
        elif spread_disp > 0:
            result["spread_edge_type"] = "lean_home"
        else:
            result["spread_edge_type"] = "lean_away"
    else:
        result["spread_disparity"] = None
        result["vegas_spread"] = None
        result["spread_edge_type"] = None

    if vegas_total is not None:
        total_disp = our_total - vegas_total
        result["total_disparity"] = round(total_disp, 1)
        result["vegas_total"] = vegas_total
        if abs(total_disp) < 2:
            result["total_edge_type"] = "neutral"
        elif total_disp > 0:
            result["total_edge_type"] = "lean_over"
        else:
            result["total_edge_type"] = "lean_under"
    else:
        result["total_disparity"] = None
        result["vegas_total"] = None
        result["total_edge_type"] = None

    # Edge score — base
    spread_score = abs(result.get("spread_disparity") or 0) * 3
    total_score = abs(result.get("total_disparity") or 0) * 1.5
    base_edge = spread_score + total_score

    # Sharp signal bonus: large sharp signal in SAME direction as our lean = stronger edge
    sharp = movement.get("sharp_signal", 0.0) if movement.get("has_movement_data") else 0.0
    steam = movement.get("steam_move", False)

    # Sharp signal in same direction as our spread edge amplifies the score
    move_dir = movement.get("move_direction")
    edge_type = result.get("spread_edge_type", "neutral")
    aligned = (
        (move_dir == "toward_home" and edge_type in ("lean_home", "fade_away")) or
        (move_dir == "toward_away" and edge_type in ("lean_away", "fade_home"))
    )
    sharp_bonus = sharp * 8 if aligned else 0
    steam_bonus = 5.0 if steam else 0.0

    result["edge_score"] = round(min(100, base_edge + sharp_bonus + steam_bonus), 1)
    result["sharp_aligned"] = aligned
    result["sharp_signal"] = round(sharp, 3)
    result["steam_move"] = steam

    if result["edge_score"] >= 15:
        result["edge_label"] = "🔥 Strong Edge"
    elif result["edge_score"] >= 8:
        result["edge_label"] = "⚡ Moderate Edge"
    elif result["edge_score"] >= 3:
        result["edge_label"] = "→ Slight Lean"
    else:
        result["edge_label"] = "— Neutral"

    return result


def _format_injury_notes(injuries: list, cascades: list, team: str) -> list[str]:
    notes = []
    for inj in injuries:
        if inj["status"] in ("out", "ir", "pup", "suspended"):
            notes.append(f"❌ {inj['name']} ({inj['position_group']}) — {inj['status'].upper()}")
        elif inj["status"] == "doubtful":
            notes.append(f"⚠ {inj['name']} ({inj['position_group']}) — Doubtful")
        elif inj["status"] == "questionable":
            notes.append(f"? {inj['name']} ({inj['position_group']}) — Questionable")
    for cascade in cascades:
        notes.append(f"↓ {cascade['starter_out']} out → {cascade['backup_in']} in ({cascade['position_group']})")
    return notes


async def generate_weekly_card(
    league: str,
    week: int,
    season: Optional[int],
    nfl_team_stats: dict,
    cfb_sp_lookup: dict,
) -> dict:
    """
    Generate the full weekly card with all adjustments applied.
    """
    league_upper = league.upper()
    logger.info(f"Generating {league_upper} Week {week} card...")

    # 1. Fetch schedule
    if league_upper == "NFL":
        games = await get_nfl_schedule_upcoming(week=week, season=season)
    else:
        games = await get_cfb_upcoming(week=week, season=season)

    if not games:
        return {"league": league_upper, "week": week, "games": [], "error": "No games found"}

    # 2. Fetch lines
    lines = await get_lines(league=league_upper, week=week)
    lines_lookup = build_lines_lookup(lines)

    # 3. Take a fresh snapshot for movement tracking
    await snapshotter.take_snapshot(league_upper)

    card_games = []
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        neutral = game.get("neutral_site", False)
        if not home or not away:
            continue

        # 4. Base features
        if league_upper == "NFL":
            features = build_nfl_matchup_features(home, away, nfl_team_stats)
        else:
            features = build_cfb_matchup_features(home, away, cfb_sp_lookup, neutral_site=neutral)

        # 5. Apply all adjustments (roster + injury + sharp money)
        features, adj_summary = apply_all_adjustments(features, home, away, league_upper)

        # 6. Predict
        if league_upper == "NFL":
            prediction = predictor.predict_nfl(features)
        else:
            prediction = predictor.predict_cfb(features)

        # 7. Find line + calculate disparity
        line = find_line(lines_lookup, home, away)
        movement = adj_summary["movement"]
        disparity = calculate_disparity(prediction, line, movement)

        # 8. Format human-readable notes
        home_injury_notes = _format_injury_notes(
            adj_summary["home_injuries"], adj_summary["home_cascades"], home
        )
        away_injury_notes = _format_injury_notes(
            adj_summary["away_injuries"], adj_summary["away_cascades"], away
        )

        roster_notes = []
        if abs(adj_summary["home_roster_adj"]) >= 0.5:
            d = "↑" if adj_summary["home_roster_adj"] > 0 else "↓"
            roster_notes.append(f"{d} {home} roster adj: {adj_summary['home_roster_adj']:+.1f} pts")
        if abs(adj_summary["away_roster_adj"]) >= 0.5:
            d = "↑" if adj_summary["away_roster_adj"] > 0 else "↓"
            roster_notes.append(f"{d} {away} roster adj: {adj_summary['away_roster_adj']:+.1f} pts")

        # Movement summary for display
        mv_display = None
        if movement.get("has_movement_data") and movement.get("spread_move") is not None:
            mv = movement["spread_move"]
            mv_display = {
                "spread_open": movement.get("spread_open"),
                "spread_current": movement.get("spread_current"),
                "spread_move": mv,
                "total_open": movement.get("total_open"),
                "total_current": movement.get("total_current"),
                "total_move": movement.get("total_move"),
                "sharp_signal": movement.get("sharp_signal", 0),
                "steam_move": movement.get("steam_move", False),
                "move_direction": movement.get("move_direction"),
                "hours_tracked": movement.get("hours_elapsed", 0),
            }

        card_games.append({
            "home_team": home,
            "away_team": away,
            "date": game.get("date", game.get("start_date", "")),
            "venue": game.get("venue", ""),
            "prediction": prediction,
            "disparity": disparity,
            "line_movement": mv_display,
            "home_injury_notes": home_injury_notes,
            "away_injury_notes": away_injury_notes,
            "roster_notes": roster_notes,
            "adjustments": {
                "home_roster": adj_summary["home_roster_adj"],
                "away_roster": adj_summary["away_roster_adj"],
                "home_injury": adj_summary["home_injury_adj"],
                "away_injury": adj_summary["away_injury_adj"],
            },
            "league": league_upper,
            "week": week,
        })

    # Sort by edge_score
    card_games.sort(key=lambda g: g["disparity"].get("edge_score") or 0, reverse=True)

    lines_coverage = sum(1 for g in card_games if g["disparity"]["has_line"])
    games_with_movement = sum(1 for g in card_games if g["line_movement"] is not None)

    return {
        "league": league_upper,
        "week": week,
        "season": season,
        "generated_at": datetime.utcnow().isoformat(),
        "total_games": len(card_games),
        "games_with_lines": lines_coverage,
        "games_with_movement": games_with_movement,
        "snapshot_stats": snapshotter.get_snapshot_stats(),
        "games": card_games,
    }
