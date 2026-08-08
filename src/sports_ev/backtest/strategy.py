from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy.orm import Session

from sports_ev.backtest.models import BetCandidate, BetSide
from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.pipeline import FeaturePipeline
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.pricing import DevigMethod, devig_two_way, expected_value, is_plus_ev


class BetStrategy(Protocol):
    name: str

    def generate_candidates(
        self, session: Session, game: Game, settings: Settings
    ) -> list[BetCandidate]: ...


@dataclass
class MarketDivergenceStrategy:
    """
    Baseline walk-forward strategy (pre-model):
    Bet when Pinnacle no-vig probability beats soft-book implied odds by min edge.
    Uses only pre-kickoff snapshots via FeaturePipeline data loaders.
    """

    name: str = "market_divergence_v1"
    soft_book: str = "draftkings"
    market_type: str = "spread"
    devig_method: DevigMethod = DevigMethod.MULTIPLICATIVE

    def generate_candidates(
        self, session: Session, game: Game, settings: Settings | None = None
    ) -> list[BetCandidate]:
        settings = settings or get_settings()
        loader = FeatureDataLoader(session)
        ctx = FeatureContext.for_game(game)

        pinnacle = loader.latest_odds_snapshot(
            ctx, book=settings.pinnacle_bookmaker, market_type=self.market_type
        )
        soft = loader.latest_odds_snapshot(ctx, book=self.soft_book, market_type=self.market_type)

        if pinnacle is None or soft is None:
            return []
        if (
            pinnacle.odds_home is None
            or pinnacle.odds_away is None
            or soft.odds_home is None
            or soft.odds_away is None
        ):
            return []

        if self.market_type == "spread" and soft.line is None:
            return []

        fair_home, fair_away = devig_two_way(
            pinnacle.odds_home,
            pinnacle.odds_away,
            method=self.devig_method,
        )

        candidates: list[BetCandidate] = []
        decision_time = soft.captured_at
        line = soft.line if self.market_type == "spread" else None

        home_ev = expected_value(fair_home, soft.odds_home, stake=1.0)
        if is_plus_ev(fair_home, soft.odds_home, min_edge_pct=settings.min_edge_pct):
            candidates.append(
                BetCandidate(
                    game_id=game.game_id,
                    book=self.soft_book,
                    side=BetSide.HOME,
                    american_odds=soft.odds_home,
                    line=line,
                    true_prob=fair_home,
                    edge_pct=home_ev.edge_pct,
                    decision_time=decision_time,
                    market_type=self.market_type,
                )
            )

        away_ev = expected_value(fair_away, soft.odds_away, stake=1.0)
        if is_plus_ev(fair_away, soft.odds_away, min_edge_pct=settings.min_edge_pct):
            candidates.append(
                BetCandidate(
                    game_id=game.game_id,
                    book=self.soft_book,
                    side=BetSide.AWAY,
                    american_odds=soft.odds_away,
                    line=line,
                    true_prob=fair_away,
                    edge_pct=away_ev.edge_pct,
                    decision_time=decision_time,
                    market_type=self.market_type,
                )
            )

        if len(candidates) <= 1:
            return candidates
        return [max(candidates, key=lambda c: c.edge_pct)]
