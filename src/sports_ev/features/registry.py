from __future__ import annotations

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.features.mlb.pipeline import MlbFeaturePipeline
from sports_ev.features.pipeline import FeaturePipeline, FeatureVector


def get_feature_pipeline(
    session: Session,
    *,
    sport: str,
    settings: Settings | None = None,
) -> FeaturePipeline | MlbFeaturePipeline:
    if sport.lower() == "mlb":
        return MlbFeaturePipeline(session, settings or get_settings())
    return FeaturePipeline(session, settings or get_settings())


def build_features_for_game(
    session: Session,
    game_id: str,
    *,
    validate_leakage: bool = True,
    settings: Settings | None = None,
) -> FeatureVector:
    from sports_ev.db.models import Game

    game = session.query(Game).filter_by(game_id=game_id).one()
    pipeline = get_feature_pipeline(session, sport=game.sport, settings=settings)
    return pipeline.build_for_game(game_id, validate_leakage=validate_leakage)
