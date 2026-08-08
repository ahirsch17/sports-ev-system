from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from sports_ev.models.imputation import blend_with_market, clip_probability, impute_features
from sports_ev.models.mlb_version import mlb_model_version

DEFAULT_PROB_MIN = 0.10
DEFAULT_PROB_MAX = 0.90


@dataclass
class MlbPredictionResult:
    home_win_prob: float
    away_win_prob: float
    raw_prob: float
    calibrated_prob: float
    missing_feature_ratio: float
    shap_values: dict[str, float]
    feature_snapshot: dict[str, float | None]


@dataclass
class MlbMoneylineModel:
    booster: lgb.Booster
    calibrator: IsotonicRegression
    feature_names: list[str]
    feature_imputations: dict[str, float] = field(default_factory=dict)
    prob_min: float = DEFAULT_PROB_MIN
    prob_max: float = DEFAULT_PROB_MAX
    model_version: str = field(default_factory=mlb_model_version)

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x)

    def predict_calibrated(self, x: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(x)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def predict_one(
        self, features: dict[str, float | None], *, compute_shap: bool = True
    ) -> MlbPredictionResult:
        from sports_ev.models.dataset import row_to_vector
        from sports_ev.models.explain import compute_shap_values

        imputed, missing_ratio = impute_features(
            features,
            self.feature_names,
            self.feature_imputations,
        )
        x = row_to_vector(imputed, self.feature_names)
        raw = float(self.predict_raw(x)[0])
        calibrated = float(self.predict_calibrated(x)[0])
        calibrated = clip_probability(calibrated, low=self.prob_min, high=self.prob_max)

        market_fair_home = features.get("pinnacle_fair_prob_home")
        if market_fair_home is not None:
            calibrated = blend_with_market(
                calibrated,
                float(market_fair_home),
                missing_ratio=missing_ratio,
            )
        calibrated = clip_probability(calibrated, low=self.prob_min, high=self.prob_max)

        shap = compute_shap_values(self, x[0]) if compute_shap else {}

        return MlbPredictionResult(
            home_win_prob=calibrated,
            away_win_prob=1.0 - calibrated,
            raw_prob=raw,
            calibrated_prob=calibrated,
            missing_feature_ratio=missing_ratio,
            shap_values=shap,
            feature_snapshot=features,
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "booster": self.booster,
                "calibrator": self.calibrator,
                "feature_names": self.feature_names,
                "feature_imputations": self.feature_imputations,
                "prob_min": self.prob_min,
                "prob_max": self.prob_max,
                "model_version": self.model_version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> MlbMoneylineModel:
        payload = joblib.load(path)
        return cls(
            booster=payload["booster"],
            calibrator=payload["calibrator"],
            feature_names=payload["feature_names"],
            feature_imputations=payload.get("feature_imputations", {}),
            prob_min=payload.get("prob_min", DEFAULT_PROB_MIN),
            prob_max=payload.get("prob_max", DEFAULT_PROB_MAX),
            model_version=payload.get("model_version", mlb_model_version()),
        )
