from __future__ import annotations

import numpy as np


def compute_feature_imputations(
    rows: list,
    feature_names: list[str],
) -> dict[str, float]:
    imputations: dict[str, float] = {}
    for name in feature_names:
        values = [
            float(row.features[name])
            for row in rows
            if row.features.get(name) is not None
        ]
        imputations[name] = float(np.median(values)) if values else 0.0
    return imputations


def impute_features(
    features: dict[str, float | None],
    feature_names: list[str],
    imputations: dict[str, float],
) -> tuple[dict[str, float], float]:
    imputed: dict[str, float] = {}
    missing = 0
    for name in feature_names:
        value = features.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            imputed[name] = imputations.get(name, 0.0)
            missing += 1
        else:
            imputed[name] = float(value)
    missing_ratio = missing / len(feature_names) if feature_names else 0.0
    return imputed, missing_ratio


def blend_with_market(
    model_prob: float,
    market_prob: float | None,
    *,
    missing_ratio: float,
    min_model_weight: float = 0.25,
    base_model_weight: float = 0.55,
) -> float:
    if market_prob is None:
        return model_prob
    model_weight = max(min_model_weight, base_model_weight - (missing_ratio * 0.45))
    return (model_weight * model_prob) + ((1.0 - model_weight) * market_prob)


def clip_probability(prob: float, *, low: float = 0.10, high: float = 0.90) -> float:
    return float(min(high, max(low, prob)))
