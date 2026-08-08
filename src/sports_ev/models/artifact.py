from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from sports_ev.models.version import model_version


@dataclass
class PredictionResult:
    home_cover_prob: float
    away_cover_prob: float
    raw_prob: float
    shap_values: dict[str, float]
    feature_snapshot: dict[str, float | None]


@dataclass
class NflSpreadModel:
    booster: lgb.Booster
    calibrator: IsotonicRegression
    feature_names: list[str]
    model_version: str = field(default_factory=model_version)

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x)

    def predict_calibrated(self, x: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(x)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def predict_one(self, features: dict[str, float | None], *, compute_shap: bool = True) -> PredictionResult:
        from sports_ev.models.dataset import row_to_vector
        from sports_ev.models.explain import compute_shap_values

        x = row_to_vector(features, self.feature_names)
        raw = float(self.predict_raw(x)[0])
        calibrated = float(self.predict_calibrated(x)[0])
        shap = compute_shap_values(self, x[0]) if compute_shap else {}

        return PredictionResult(
            home_cover_prob=calibrated,
            away_cover_prob=1.0 - calibrated,
            raw_prob=raw,
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
                "model_version": self.model_version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> NflSpreadModel:
        payload = joblib.load(path)
        return cls(
            booster=payload["booster"],
            calibrator=payload["calibrator"],
            feature_names=payload["feature_names"],
            model_version=payload.get("model_version", model_version()),
        )
