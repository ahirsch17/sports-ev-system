from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:

    class ShapModel(Protocol):
        booster: object
        feature_names: list[str]


def compute_shap_values(model: ShapModel, x_row: np.ndarray) -> dict[str, float]:
    """Return SHAP values keyed by feature name. Empty dict when shap is unavailable."""
    try:
        import shap
    except ImportError:
        return {}

    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(x_row.reshape(1, -1))
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    row_values = np.asarray(values).reshape(-1)

    return {
        name: float(value)
        for name, value in zip(model.feature_names, row_values, strict=True)
        if not np.isnan(value)
    }
