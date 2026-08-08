from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.injury import build_injury_features
from sports_ev.features.leakage import assert_no_leakage
from sports_ev.features.market import build_market_features
from sports_ev.features.mlb.pitcher import build_mlb_pitcher_features
from sports_ev.features.mlb.team import (
    build_mlb_matchup_features,
    build_mlb_schedule_features,
    build_mlb_team_features,
)
from sports_ev.features.pipeline import FeatureVector
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.features.version import feature_version
from sports_ev.pricing import DevigMethod


class MlbFeaturePipeline:
    """Versioned MLB moneyline feature pipeline with leakage guards."""

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.loader = FeatureDataLoader(session)

    def build_for_game(
        self,
        game_id: str,
        *,
        devig_method: DevigMethod | None = None,
        validate_leakage: bool = True,
    ) -> FeatureVector:
        game = self.session.query(Game).filter_by(game_id=game_id).one()
        ctx = FeatureContext.for_game(game)

        features: dict[str, float | None] = {}
        features.update(
            build_mlb_team_features(
                self.loader,
                ctx,
                team=game.home_team,
                opponent=game.away_team,
                side="home",
            )
        )
        features.update(
            build_mlb_team_features(
                self.loader,
                ctx,
                team=game.away_team,
                opponent=game.home_team,
                side="away",
            )
        )
        features.update(build_mlb_matchup_features(self.loader, ctx))
        features.update(build_mlb_pitcher_features(self.loader, ctx))
        features.update(build_mlb_schedule_features(ctx))
        features.update(build_injury_features(self.loader, ctx))
        features.update(
            build_market_features(
                self.loader,
                ctx,
                devig_method=devig_method,
                settings=self.settings,
                market_type="moneyline",
            )
        )

        if validate_leakage:
            assert_no_leakage(ctx.audit)

        audit_sources = [
            {
                "name": s.name,
                "value": s.value,
                "known_at": s.known_at.isoformat(),
                "source": s.source,
            }
            for s in ctx.audit.sources
        ]

        return FeatureVector(
            game_id=game_id,
            feature_version=feature_version("mlb"),
            features=features,
            audit_sources=audit_sources,
        )
