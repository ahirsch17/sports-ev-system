"""NFL spread gradient-boosted model with calibration and SHAP explainability."""

from sports_ev.models.artifact import NflSpreadModel, PredictionResult
from sports_ev.models.dataset import LabeledRow, build_labeled_rows
from sports_ev.models.split import TimeSplit, time_ordered_split
from sports_ev.models.storage import save_prediction
from sports_ev.models.strategy import ModelSpreadStrategy, WalkForwardModelStrategy
from sports_ev.models.train import TrainConfig, TrainMetrics, train_and_save, train_from_rows
from sports_ev.models.version import MODEL_VERSION, model_version

__all__ = [
    "MODEL_VERSION",
    "LabeledRow",
    "ModelSpreadStrategy",
    "NflSpreadModel",
    "PredictionResult",
    "TimeSplit",
    "TrainConfig",
    "TrainMetrics",
    "WalkForwardModelStrategy",
    "build_labeled_rows",
    "model_version",
    "save_prediction",
    "time_ordered_split",
    "train_and_save",
    "train_from_rows",
]
