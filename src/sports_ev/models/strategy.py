from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from sports_ev.backtest.models import BetCandidate, BetSide
from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.pipeline import FeaturePipeline
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.models.artifact import NflSpreadModel
from sports_ev.models.dataset import build_labeled_rows
from sports_ev.models.train import TrainConfig, train_from_rows
from sports_ev.pricing import expected_value, is_plus_ev


@dataclass
class ModelSpreadStrategy:
    """Backtest strategy using a pre-trained NFL spread model artifact."""

    model: NflSpreadModel
    name: str = "nfl_spread_model_v1"
    soft_book: str = "draftkings"

    def generate_candidates(
        self, session: Session, game: Game, settings: Settings | None = None
    ) -> list[BetCandidate]:
        settings = settings or get_settings()
        pipeline = FeaturePipeline(session, settings)
        loader = FeatureDataLoader(session)
        ctx = FeatureContext.for_game(game)

        soft = loader.latest_odds_snapshot(ctx, book=self.soft_book, market_type="spread")
        if soft is None or soft.odds_home is None or soft.odds_away is None or soft.line is None:
            return []

        vector = pipeline.build_for_game(game.game_id, validate_leakage=True)
        prediction = self.model.predict_one(vector.features, compute_shap=False)

        candidates: list[BetCandidate] = []
        decision_time = soft.captured_at

        home_ev = expected_value(prediction.home_cover_prob, soft.odds_home, stake=1.0)
        if is_plus_ev(
            prediction.home_cover_prob,
            soft.odds_home,
            min_edge_pct=settings.min_edge_pct,
        ):
            candidates.append(
                BetCandidate(
                    game_id=game.game_id,
                    book=self.soft_book,
                    side=BetSide.HOME,
                    american_odds=soft.odds_home,
                    line=soft.line,
                    true_prob=prediction.home_cover_prob,
                    edge_pct=home_ev.edge_pct,
                    decision_time=decision_time,
                    market_type="spread",
                )
            )

        away_ev = expected_value(prediction.away_cover_prob, soft.odds_away, stake=1.0)
        if is_plus_ev(
            prediction.away_cover_prob,
            soft.odds_away,
            min_edge_pct=settings.min_edge_pct,
        ):
            candidates.append(
                BetCandidate(
                    game_id=game.game_id,
                    book=self.soft_book,
                    side=BetSide.AWAY,
                    american_odds=soft.odds_away,
                    line=soft.line,
                    true_prob=prediction.away_cover_prob,
                    edge_pct=away_ev.edge_pct,
                    decision_time=decision_time,
                    market_type="spread",
                )
            )

        if len(candidates) <= 1:
            return candidates
        return [max(candidates, key=lambda c: c.edge_pct)]


@dataclass
class WalkForwardModelStrategy:
    """
    Walk-forward model strategy: retrains on all settled games before each decision.
    Only uses information available before the current game's kickoff.
    """

    name: str = "nfl_spread_walk_forward_v1"
    soft_book: str = "draftkings"
    min_train_games: int = 8
    train_config: TrainConfig | None = None

    def generate_candidates(
        self, session: Session, game: Game, settings: Settings | None = None
    ) -> list[BetCandidate]:
        settings = settings or get_settings()
        train_rows = build_labeled_rows(
            session,
            sport=game.sport,
            before_kickoff=game.kickoff_time,
            settings=settings,
        )
        if len(train_rows) < self.min_train_games:
            return []

        config = self.train_config or TrainConfig(num_boost_round=30, min_data_in_leaf=1)
        model, _ = train_from_rows(train_rows, config=config)
        strategy = ModelSpreadStrategy(model=model, soft_book=self.soft_book, name=self.name)
        return strategy.generate_candidates(session, game, settings)
