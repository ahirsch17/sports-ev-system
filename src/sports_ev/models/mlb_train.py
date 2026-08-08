from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression

from sports_ev.models.imputation import compute_feature_imputations
from sports_ev.models.mlb_artifact import MlbMoneylineModel
from sports_ev.models.mlb_dataset import MlbLabeledRow
from sports_ev.models.mlb_version import mlb_model_version
from sports_ev.models.dataset import feature_names_from_rows, rows_to_matrix
from sports_ev.models.split import time_ordered_split
from sports_ev.models.train import TrainConfig, TrainMetrics, _lgb_params


@dataclass
class MlbTrainConfig(TrainConfig):
    val_fraction: float = 0.2
    num_boost_round: int = 120
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 25
    random_state: int = 42


def _rows_to_matrix(rows: list[MlbLabeledRow], feature_names: list[str]):
    from sports_ev.models.dataset import LabeledRow

    converted = [
        LabeledRow(
            game_id=r.game_id,
            kickoff_time=r.kickoff_time,
            features=r.features,
            label=r.label,
            spread_line=0.0,
        )
        for r in rows
    ]
    return rows_to_matrix(converted, feature_names)


def train_mlb_from_rows(
    rows: list[MlbLabeledRow], config: TrainConfig | None = None
) -> tuple[MlbMoneylineModel, TrainMetrics]:
    config = config or MlbTrainConfig()
    if not rows:
        raise ValueError("Cannot train without labeled rows")

    from sports_ev.models.dataset import LabeledRow

    converted = [
        LabeledRow(
            game_id=r.game_id,
            kickoff_time=r.kickoff_time,
            features=r.features,
            label=r.label,
            spread_line=0.0,
        )
        for r in rows
    ]
    split = time_ordered_split(converted, val_fraction=config.val_fraction)
    feature_names = feature_names_from_rows(converted)
    feature_imputations = compute_feature_imputations(split.train, feature_names)

    def _impute_row(row: LabeledRow) -> LabeledRow:
        imputed = {
            name: (
                float(row.features[name])
                if row.features.get(name) is not None
                else feature_imputations[name]
            )
            for name in feature_names
        }
        return LabeledRow(
            game_id=row.game_id,
            kickoff_time=row.kickoff_time,
            features=imputed,
            label=row.label,
            spread_line=row.spread_line,
        )

    train_rows = [_impute_row(row) for row in split.train]
    x_train, y_train = rows_to_matrix(train_rows, feature_names)

    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(
        _lgb_params(config),
        train_set,
        num_boost_round=config.num_boost_round,
    )

    calibrator = IsotonicRegression(out_of_bounds="clip")
    raw_val_brier = None
    calibrated_val_brier = None
    if split.validation:
        val_rows = [_impute_row(row) for row in split.validation]
        x_val, y_val = rows_to_matrix(val_rows, feature_names)
        raw_val = booster.predict(x_val)
        calibrator.fit(raw_val, y_val)
        calibrated_val = np.clip(calibrator.predict(raw_val), 0.0, 1.0)
        raw_val_brier = float(brier_score_loss(y_val, raw_val))
        calibrated_val_brier = float(brier_score_loss(y_val, calibrated_val))
    else:
        raw_train = booster.predict(x_train)
        calibrator.fit(raw_train, y_train)

    model = MlbMoneylineModel(
        booster=booster,
        calibrator=calibrator,
        feature_names=feature_names,
        feature_imputations=feature_imputations,
        model_version=mlb_model_version(),
    )
    metrics = TrainMetrics(
        train_rows=len(split.train),
        validation_rows=len(split.validation),
        raw_val_brier=raw_val_brier,
        calibrated_val_brier=calibrated_val_brier,
    )
    return model, metrics


def train_mlb_and_save(
    rows: list[MlbLabeledRow], artifact_path: Path | str, config: TrainConfig | None = None
):
    model, metrics = train_mlb_from_rows(rows, config=config)
    model.save(artifact_path)
    return model, metrics
