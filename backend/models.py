"""
models.py
Margin and total prediction models.
Uses Ridge regression + Gradient Boosting ensemble.
Trains on historical data fetched at startup.
"""

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(MODEL_DIR, exist_ok=True)

NFL_FEATURE_COLS = [
    "home_off_avg", "home_def_avg", "away_off_avg", "away_def_avg",
    "off_delta", "def_delta", "home_margin_avg", "away_margin_avg",
    "home_field_advantage", "combined_off", "combined_def",
    "home_roster_adj", "away_roster_adj",
    "home_injury_adj", "away_injury_adj",
    "sharp_signal", "spread_move", "total_move", "move_velocity", "steam_move",
]

CFB_FEATURE_COLS = [
    "sp_diff", "home_sp_overall", "away_sp_overall",
    "home_sp_offense", "home_sp_defense", "away_sp_offense", "away_sp_defense",
    "off_def_matchup_home", "off_def_matchup_away",
    "home_field_advantage", "predicted_home_off_contribution", "predicted_away_off_contribution",
    "home_roster_adj", "away_roster_adj",
    "home_injury_adj", "away_injury_adj",
    "sharp_signal", "spread_move", "total_move", "move_velocity", "steam_move",
]


def _make_ensemble(target: str) -> dict:
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "gbm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ))
        ]),
    }


class GridironPredictor:
    def __init__(self):
        self.nfl_margin_models = None
        self.nfl_total_models = None
        self.cfb_margin_models = None
        self.cfb_total_models = None
        self.nfl_trained = False
        self.cfb_trained = False
        self.nfl_margin_cv = None
        self.nfl_total_cv = None
        self.cfb_margin_cv = None
        self.cfb_total_cv = None

    def train_nfl(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 20:
            logger.warning("Not enough NFL data to train — using baseline model")
            self.nfl_trained = False
            return {"error": "Insufficient data", "n_samples": len(df)}

        for col in NFL_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0

        X = df[NFL_FEATURE_COLS].fillna(0).values
        y_margin = df["margin"].values
        y_total = df["total"].values

        self.nfl_margin_models = _make_ensemble("margin")
        self.nfl_total_models = _make_ensemble("total")

        for pipe in self.nfl_margin_models.values():
            pipe.fit(X, y_margin)

        for pipe in self.nfl_total_models.values():
            pipe.fit(X, y_total)

        ridge_margin_cv = cross_val_score(
            self.nfl_margin_models["ridge"], X, y_margin, cv=5, scoring="r2"
        ).mean()

        ridge_total_cv = cross_val_score(
            self.nfl_total_models["ridge"], X, y_total, cv=5, scoring="r2"
        ).mean()

        self.nfl_margin_cv = ridge_margin_cv
        self.nfl_total_cv = ridge_total_cv
        self.nfl_trained = True

        logger.info(
            f"NFL trained on {len(df)} games. "
            f"Margin R²={ridge_margin_cv:.3f}, Total R²={ridge_total_cv:.3f}"
        )

        return {
            "n_samples": len(df),
            "margin_r2": round(ridge_margin_cv, 3),
            "total_r2": round(ridge_total_cv, 3),
        }

    def train_cfb(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 30:
            logger.warning("Not enough CFB data to train")
            self.cfb_trained = False
            return {"error": "Insufficient data", "n_samples": len(df)}

        for col in CFB_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0

        X = df[CFB_FEATURE_COLS].fillna(0).values
        y_margin = df["margin"].values
        y_total = df["total"].values

        self.cfb_margin_models = _make_ensemble("margin")
        self.cfb_total_models = _make_ensemble("total")

        for pipe in self.cfb_margin_models.values():
            pipe.fit(X, y_margin)

        for pipe in self.cfb_total_models.values():
            pipe.fit(X, y_total)

        ridge_margin_cv = cross_val_score(
            self.cfb_margin_models["ridge"], X, y_margin, cv=5, scoring="r2"
        ).mean()

        ridge_total_cv = cross_val_score(
            self.cfb_total_models["ridge"], X, y_total, cv=5, scoring="r2"
        ).mean()

        self.cfb_margin_cv = ridge_margin_cv
        self.cfb_total_cv = ridge_total_cv
        self.cfb_trained = True

        logger.info(
            f"CFB trained on {len(df)} games. "
            f"Margin R²={ridge_margin_cv:.3f}, Total R²={ridge_total_cv:.3f}"
        )

        return {
            "n_samples": len(df),
            "margin_r2": round(ridge_margin_cv, 3),
            "total_r2": round(ridge_total_cv, 3),
        }

    def _predict_ensemble(self, models: dict, X: np.ndarray) -> float:
        ridge_pred = models["ridge"].predict(X)[0]
        gbm_pred = models["gbm"].predict(X)[0]
        return 0.4 * ridge_pred + 0.6 * gbm_pred

    def predict_nfl(self, features: dict) -> dict:
        X = np.array([[features.get(c, 0.0) for c in NFL_FEATURE_COLS]])

        if not self.nfl_trained or self.nfl_margin_models is None:
            margin = (
                features.get("home_margin_avg", 0)
                - features.get("away_margin_avg", 0)
                + features.get("home_field_advantage", 2.5)
            )
            total = features.get("combined_off", 46.0)
        else:
            margin = self._predict_ensemble(self.nfl_margin_models, X)
            total = self._predict_ensemble(self.nfl_total_models, X)

        return self._format_prediction(margin, total, "NFL")

    def predict_cfb(self, features: dict) -> dict:
        X = np.array([[features.get(c, 0.0) for c in CFB_FEATURE_COLS]])

        if not self.cfb_trained or self.cfb_margin_models is None:
            sp_diff = features.get("sp_diff", 0.0)
            margin = sp_diff * 0.6 + features.get("home_field_advantage", 3.0)
            total = (
                55.0
                + features.get("off_def_matchup_home", 0) * 0.3
                + features.get("off_def_matchup_away", 0) * 0.3
            )
        else:
            margin = self._predict_ensemble(self.cfb_margin_models, X)
            total = self._predict_ensemble(self.cfb_total_models, X)

        return self._format_prediction(margin, total, "CFB")

    def _format_prediction(self, margin: float, total: float, league: str) -> dict:
        margin_rmse = 9.0 if league == "NFL" else 14.0
        total_rmse = 12.0 if league == "NFL" else 18.0

        home_score = (total + margin) / 2
        away_score = (total - margin) / 2

        home_score = max(0, round(home_score, 1))
        away_score = max(0, round(away_score, 1))
        margin = round(margin, 1)
        total = round(total, 1)

        margin_lo = round(margin - 1.28 * margin_rmse, 1)
        margin_hi = round(margin + 1.28 * margin_rmse, 1)
        total_lo = round(total - 1.28 * total_rmse, 1)
        total_hi = round(total + 1.28 * total_rmse, 1)

        return {
            "predicted_home_score": home_score,
            "predicted_away_score": away_score,
            "predicted_margin": margin,
            "predicted_total": total,
            "margin_80_lo": margin_lo,
            "margin_80_hi": margin_hi,
            "total_80_lo": total_lo,
            "total_80_hi": total_hi,
            "home_win_prob": round(self._margin_to_prob(margin, margin_rmse), 3),
            "model_trained": self.nfl_trained if league == "NFL" else self.cfb_trained,
        }

    def _margin_to_prob(self, margin: float, rmse: float) -> float:
        from scipy.stats import norm
        return float(norm.cdf(margin / rmse))

    def status(self) -> dict:
        return {
            "nfl_trained": self.nfl_trained,
            "cfb_trained": self.cfb_trained,
            "nfl_margin_r2": self.nfl_margin_cv,
            "nfl_total_r2": self.nfl_total_cv,
            "cfb_margin_r2": self.cfb_margin_cv,
            "cfb_total_r2": self.cfb_total_cv,
        }


predictor = GridironPredictor()