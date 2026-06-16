"""
feature_engine.py
Transforms raw game/team data into model-ready features.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# NFL Feature Builder
# ─────────────────────────────────────────────

def build_nfl_game_features(games: list[dict]) -> pd.DataFrame:
    """
    From a list of completed NFL games, build rolling team efficiency features.
    Returns one row per game with home/away stat deltas.
    """
    df = pd.DataFrame(games)
    if df.empty:
        return df

    df["margin"] = df["home_score"] - df["away_score"]
    df["total"] = df["home_score"] + df["away_score"]

    # Rolling 4-game averages per team
    records = []
    for _, row in df.iterrows():
        records.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "margin": row["margin"],
            "total": row["total"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
        })

    return pd.DataFrame(records)


def build_nfl_team_rolling(games: list[dict], window: int = 6) -> dict:
    """
    Returns a dict keyed by team name with rolling offensive/defensive averages.
    """
    df = pd.DataFrame(games)
    if df.empty:
        return {}

    team_stats = {}

    all_teams = set(df["home_team"].tolist() + df["away_team"].tolist())

    for team in all_teams:
        home_games = df[df["home_team"] == team].copy()
        away_games = df[df["away_team"] == team].copy()

        home_games["pts_for"] = home_games["home_score"]
        home_games["pts_against"] = home_games["away_score"]
        away_games["pts_for"] = away_games["away_score"]
        away_games["pts_against"] = away_games["home_score"]

        all_g = pd.concat([home_games[["pts_for", "pts_against"]],
                           away_games[["pts_for", "pts_against"]]]).sort_index()

        if len(all_g) == 0:
            continue

        recent = all_g.tail(window)
        team_stats[team] = {
            "avg_pts_for": round(recent["pts_for"].mean(), 2),
            "avg_pts_against": round(recent["pts_against"].mean(), 2),
            "avg_margin": round((recent["pts_for"] - recent["pts_against"]).mean(), 2),
        }

    return team_stats


def build_nfl_matchup_features(
    home_team: str,
    away_team: str,
    team_stats: dict,
    neutral_site: bool = False
) -> dict:
    """
    Returns a flat feature dict for a single matchup prediction.
    """
    home = team_stats.get(home_team, {"avg_pts_for": 23.0, "avg_pts_against": 23.0, "avg_margin": 0.0})
    away = team_stats.get(away_team, {"avg_pts_for": 23.0, "avg_pts_against": 23.0, "avg_margin": 0.0})

    return {
        "home_off_avg": home["avg_pts_for"],
        "home_def_avg": home["avg_pts_against"],
        "away_off_avg": away["avg_pts_for"],
        "away_def_avg": away["avg_pts_against"],
        "off_delta": home["avg_pts_for"] - away["avg_pts_for"],
        "def_delta": home["avg_pts_against"] - away["avg_pts_against"],
        "home_margin_avg": home["avg_margin"],
        "away_margin_avg": away["avg_margin"],
        "home_field_advantage": 0.0 if neutral_site else 2.5,  # NFL HFA ~2.5 pts
        "combined_off": home["avg_pts_for"] + away["avg_pts_for"],
        "combined_def": home["avg_pts_against"] + away["avg_pts_against"],
    }


# ─────────────────────────────────────────────
# CFB Feature Builder
# ─────────────────────────────────────────────

def build_cfb_sp_lookup(sp_ratings: list[dict]) -> dict:
    """
    Returns a dict keyed by team name with SP+ offense/defense/overall.
    SP+ is the single best predictor in college football.
    """
    lookup = {}
    for entry in sp_ratings:
        team = entry.get("team", "")
        lookup[team] = {
            "sp_overall": entry.get("rating", 0.0),
            "sp_offense": entry.get("offense", {}).get("rating", 0.0) if isinstance(entry.get("offense"), dict) else 0.0,
            "sp_defense": entry.get("defense", {}).get("rating", 0.0) if isinstance(entry.get("defense"), dict) else 0.0,
        }
    return lookup


def build_cfb_matchup_features(
    home_team: str,
    away_team: str,
    sp_lookup: dict,
    neutral_site: bool = False
) -> dict:
    """
    Returns feature dict for a CFB matchup using SP+ ratings.
    """
    league_avg_sp = 0.0
    league_avg_off = 25.0
    league_avg_def = -5.0

    home = sp_lookup.get(home_team, {
        "sp_overall": league_avg_sp,
        "sp_offense": league_avg_off,
        "sp_defense": league_avg_def
    })
    away = sp_lookup.get(away_team, {
        "sp_overall": league_avg_sp,
        "sp_offense": league_avg_off,
        "sp_defense": league_avg_def
    })

    hfa = 0.0 if neutral_site else 3.0  # CFB HFA ~3 pts

    return {
        "sp_diff": home["sp_overall"] - away["sp_overall"],
        "home_sp_overall": home["sp_overall"],
        "away_sp_overall": away["sp_overall"],
        "home_sp_offense": home["sp_offense"],
        "home_sp_defense": home["sp_defense"],
        "away_sp_offense": away["sp_offense"],
        "away_sp_defense": away["sp_defense"],
        "off_def_matchup_home": home["sp_offense"] - away["sp_defense"],
        "off_def_matchup_away": away["sp_offense"] - home["sp_defense"],
        "home_field_advantage": hfa,
        "predicted_home_off_contribution": home["sp_offense"] + hfa,
        "predicted_away_off_contribution": away["sp_offense"],
    }


# ─────────────────────────────────────────────
# Training Data Builder
# ─────────────────────────────────────────────

def build_cfb_training_data(games: list[dict], sp_lookup: dict) -> pd.DataFrame:
    rows = []
    for g in games:
        if g.get("home_points") is None or g.get("away_points") is None:
            continue
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        feats = build_cfb_matchup_features(home, away, sp_lookup, neutral_site=g.get("neutral_site", False))
        feats["margin"] = g["home_points"] - g["away_points"]
        feats["total"] = g["home_points"] + g["away_points"]
        feats["home_score"] = g["home_points"]
        feats["away_score"] = g["away_points"]
        rows.append(feats)
    return pd.DataFrame(rows)
