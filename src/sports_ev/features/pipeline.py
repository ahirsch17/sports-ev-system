from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.injury import build_injury_features
from sports_ev.features.leakage import assert_no_leakage
from sports_ev.features.market import build_market_features
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.features.team import build_matchup_features, build_schedule_features, build_team_features
from sports_ev.features.version import feature_version
from sports_ev.pricing import DevigMethod


@dataclass
class FeatureVector:
    game_id: str
    feature_version: str
    features: dict[str, float | None]
    audit_sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "feature_version": self.feature_version,
            "features": self.features,
            "audit_sources": self.audit_sources,
        }


class FeaturePipeline:
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
            build_team_features(
                self.loader,
                ctx,
                team=game.home_team,
                opponent=game.away_team,
                side="home",
            )
        )
        features.update(
            build_team_features(
                self.loader,
                ctx,
                team=game.away_team,
                opponent=game.home_team,
                side="away",
            )
        )
        features.update(build_matchup_features(self.loader, ctx))
        features.update(build_injury_features(self.loader, ctx))
        features.update(build_schedule_features(self.loader, ctx))
        features.update(
            build_market_features(
                self.loader,
                ctx,
                devig_method=devig_method,
                settings=self.settings,
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
            feature_version=feature_version(game.sport),
            features=features,
            audit_sources=audit_sources,
        )
