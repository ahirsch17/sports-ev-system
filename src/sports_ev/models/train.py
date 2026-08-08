from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from sports_ev.models.artifact import NflSpreadModel
from sports_ev.models.dataset import (
    LabeledRow,
    feature_names_from_rows,
    rows_to_matrix,
)
from sports_ev.models.split import time_ordered_split
from sports_ev.models.version import model_version


@dataclass
class TrainConfig:
    val_fraction: float = 0.2
    num_boost_round: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 1
    random_state: int = 42


@dataclass
class TrainMetrics:
    train_rows: int
    validation_rows: int
    raw_val_brier: float | None
    calibrated_val_brier: float | None


def _lgb_params(config: TrainConfig) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "verbosity": -1,
        "seed": config.random_state,
        "feature_pre_filter": False,
    }


def train_from_rows(rows: list[LabeledRow], config: TrainConfig | None = None) -> tuple[NflSpreadModel, TrainMetrics]:
    config = config or TrainConfig()
    if not rows:
        raise ValueError("Cannot train without labeled rows")

    split = time_ordered_split(rows, val_fraction=config.val_fraction)
    feature_names = feature_names_from_rows(rows)
    x_train, y_train = rows_to_matrix(split.train, feature_names)

    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(
        _lgb_params(config),
        train_set,
        num_boost_round=config.num_boost_round,
    )

    calibrator = IsotonicRegression(out_of_bounds="clip")
    raw_train = booster.predict(x_train)
    calibrator.fit(raw_train, y_train)

    raw_val_brier = None
    calibrated_val_brier = None
    if split.validation:
        x_val, y_val = rows_to_matrix(split.validation, feature_names)
        raw_val = booster.predict(x_val)
        calibrated_val = np.clip(calibrator.predict(raw_val), 0.0, 1.0)
        raw_val_brier = float(brier_score_loss(y_val, raw_val))
        calibrated_val_brier = float(brier_score_loss(y_val, calibrated_val))

    model = NflSpreadModel(
        booster=booster,
        calibrator=calibrator,
        feature_names=feature_names,
        model_version=model_version(),
    )
    metrics = TrainMetrics(
        train_rows=len(split.train),
        validation_rows=len(split.validation),
        raw_val_brier=raw_val_brier,
        calibrated_val_brier=calibrated_val_brier,
    )
    return model, metrics


def train_and_save(rows: list[LabeledRow], artifact_path: Path | str, config: TrainConfig | None = None):
    model, metrics = train_from_rows(rows, config=config)
    model.save(artifact_path)
    return model, metrics
