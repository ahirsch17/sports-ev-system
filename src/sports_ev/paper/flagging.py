from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from sports_ev.backtest.models import BetSide
from sports_ev.backtest.strategy import MarketDivergenceStrategy
from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game, PaperBet
from sports_ev.features.registry import get_feature_pipeline
from sports_ev.features.version import feature_version
from sports_ev.integrity.circuit_breaker import CircuitBreaker
from sports_ev.models.artifact import NflSpreadModel
from sports_ev.models.mlb_artifact import MlbMoneylineModel
from sports_ev.models.mlb_strategy import ModelMoneylineStrategy
from sports_ev.models.strategy import ModelSpreadStrategy
from sports_ev.paper.kelly import kelly_fraction
from sports_ev.pricing import expected_value
from sports_ev.sports.registry import get_sport_config


@dataclass
class FlagResult:
    sport: str = ""
    market_type: str = ""
    games_scanned: int = 0
    opportunities_found: int = 0
    bets_flagged: int = 0
    suppressed_degraded: int = 0
    skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)


class OpportunityFlagger:
    """Scan upcoming games and log +EV paper bets with full audit trail."""

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.circuit_breaker = CircuitBreaker(session, self.settings)

    def _upcoming_games(self, *, days_ahead: int, sport: str) -> list[Game]:
        now = datetime.now(timezone.utc)
        end = now.replace(microsecond=0) + timedelta(days=days_ahead)
        return (
            self.session.query(Game)
            .filter(
                Game.sport == sport,
                Game.kickoff_time > now,
                Game.kickoff_time <= end,
            )
            .order_by(Game.kickoff_time.asc())
            .all()
        )

    def _existing_pending(
        self, game_id: str, book: str, side: str, market_type: str
    ) -> PaperBet | None:
        return (
            self.session.query(PaperBet)
            .filter(
                PaperBet.game_id == game_id,
                PaperBet.book == book,
                PaperBet.side == side,
                PaperBet.market_type == market_type,
                (PaperBet.outcome.is_(None)) | (PaperBet.outcome == "pending"),
            )
            .first()
        )

    def _build_audit_trail(
        self,
        *,
        strategy_name: str,
        model_version: str | None,
        candidate,
        feature_snapshot: dict | None,
        shap_top: dict[str, float] | None,
        sources: list[str],
        market_type: str,
    ) -> dict:
        ev = expected_value(candidate.true_prob, candidate.american_odds, stake=1.0)
        return {
            "strategy": strategy_name,
            "model_version": model_version,
            "feature_version": feature_version(),
            "market_type": market_type,
            "true_prob": candidate.true_prob,
            "american_odds": candidate.american_odds,
            "line_taken": candidate.line,
            "edge_pct": candidate.edge_pct,
            "ev_dollars": ev.ev_dollars,
            "sources": sources,
            "feature_snapshot": feature_snapshot or {},
            "shap_top": shap_top or {},
            "flagged_at": datetime.now(timezone.utc).isoformat(),
        }

    def flag_upcoming(
        self,
        *,
        sport: str = "nfl",
        days_ahead: int = 7,
        strategy: str = "market",
        soft_book: str = "draftkings",
        model_path: str | None = None,
        market_type: str | None = None,
    ) -> FlagResult:
        config = get_sport_config(sport, self.settings)
        market_type = market_type or config.default_market
        result = FlagResult(sport=config.sport, market_type=market_type)

        source_key = f"{soft_book}:{market_type}"
        if not self.circuit_breaker.is_source_trusted(source_key):
            result.suppressed_degraded += 1
            result.errors.append(f"Source {source_key} is degraded — suppressing flags")
            return result

        if strategy == "model":
            path = model_path or (
                self.settings.mlb_model_artifact_path
                if config.sport == "mlb"
                else self.settings.model_artifact_path
            )
            if not Path(path).exists():
                result.errors.append(f"Model artifact not found: {path}")
                return result

            if config.sport == "mlb":
                model = MlbMoneylineModel.load(path)
                bet_strategy = ModelMoneylineStrategy(model=model, soft_book=soft_book)
                model_version = model.model_version
                strategy_name = bet_strategy.name
            elif config.sport == "nfl" and market_type == "spread":
                model = NflSpreadModel.load(path)
                bet_strategy = ModelSpreadStrategy(model=model, soft_book=soft_book)
                model_version = model.model_version
                strategy_name = bet_strategy.name
            else:
                result.errors.append(f"Model strategy unsupported for {config.sport}/{market_type}")
                return result
        else:
            bet_strategy = MarketDivergenceStrategy(
                soft_book=soft_book,
                market_type=market_type,
            )
            model_version = None
            strategy_name = bet_strategy.name

        pipeline = get_feature_pipeline(self.session, sport=config.sport, settings=self.settings)
        games = self._upcoming_games(days_ahead=days_ahead, sport=config.sport)
        result.games_scanned = len(games)

        for game in games:
            candidates = bet_strategy.generate_candidates(self.session, game, self.settings)
            if not candidates:
                continue

            result.opportunities_found += len(candidates)
            vector = None
            if strategy == "model":
                vector = pipeline.build_for_game(game.game_id, validate_leakage=True)

            shap_top: dict[str, float] | None = None
            if strategy == "model" and config.sport == "mlb" and isinstance(bet_strategy, ModelMoneylineStrategy) and vector:
                prediction = bet_strategy.model.predict_one(vector.features, compute_shap=False)
                shap_top = None
            elif strategy == "model" and config.sport == "nfl" and isinstance(bet_strategy, ModelSpreadStrategy) and vector:
                prediction = bet_strategy.model.predict_one(vector.features, compute_shap=False)
                shap_top = None

            for candidate in candidates:
                side = candidate.side.value if isinstance(candidate.side, BetSide) else str(candidate.side)
                if self._existing_pending(game.game_id, candidate.book, side, market_type):
                    result.skipped_existing += 1
                    continue

                stake_pct = kelly_fraction(
                    candidate.true_prob,
                    candidate.american_odds,
                    fraction=self.settings.kelly_fraction,
                    max_stake_pct=self.settings.max_stake_pct,
                )
                sources = [
                    f"{self.settings.pinnacle_bookmaker}:scraped",
                    f"{candidate.book}:scraped",
                ]
                audit = self._build_audit_trail(
                    strategy_name=strategy_name,
                    model_version=model_version,
                    candidate=candidate,
                    feature_snapshot=vector.features if vector else {},
                    shap_top=shap_top,
                    sources=sources,
                    market_type=market_type,
                )

                self.session.add(
                    PaperBet(
                        game_id=game.game_id,
                        book=candidate.book,
                        market_type=market_type,
                        side=side,
                        flagged_at=candidate.decision_time,
                        suggested_stake_pct=stake_pct,
                        odds_taken=candidate.american_odds,
                        line_taken=candidate.line,
                        model_version=model_version or strategy_name,
                        edge_pct=candidate.edge_pct,
                        audit_trail=audit,
                        outcome="pending",
                    )
                )
                result.bets_flagged += 1

        return result
