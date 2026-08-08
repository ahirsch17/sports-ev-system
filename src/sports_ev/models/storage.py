from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from sports_ev.db.models import ModelPrediction
from sports_ev.models.artifact import PredictionResult
from sports_ev.models.mlb_artifact import MlbPredictionResult


def save_prediction(
    session: Session,
    *,
    game_id: str,
    market_type: str,
    model_version: str,
    result: PredictionResult,
    predicted_at: datetime | None = None,
) -> ModelPrediction:
    predicted_at = predicted_at or datetime.now(timezone.utc)
    row = ModelPrediction(
        game_id=game_id,
        model_version=model_version,
        market_type=market_type,
        predicted_prob=result.home_cover_prob,
        predicted_at=predicted_at,
        feature_snapshot=result.feature_snapshot,
        shap_values=result.shap_values or None,
    )
    session.add(row)
    session.flush()
    return row


def save_mlb_prediction(
    session: Session,
    *,
    game_id: str,
    market_type: str,
    model_version: str,
    result: MlbPredictionResult,
    predicted_at: datetime | None = None,
) -> ModelPrediction:
    predicted_at = predicted_at or datetime.now(timezone.utc)
    row = ModelPrediction(
        game_id=game_id,
        model_version=model_version,
        market_type=market_type,
        predicted_prob=result.home_win_prob,
        predicted_at=predicted_at,
        feature_snapshot=result.feature_snapshot,
        shap_values=result.shap_values or None,
    )
    session.add(row)
    session.flush()
    return row
