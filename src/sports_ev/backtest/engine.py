from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from sports_ev.backtest.models import BetCandidate, SettledBet
from sports_ev.backtest.report import build_report
from sports_ev.backtest.settlement import settle_moneyline_bet, settle_spread_bet
from sports_ev.backtest.strategy import BetStrategy, MarketDivergenceStrategy
from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.features.time_utils import ensure_utc
from sports_ev.sports.registry import get_sport_config


@dataclass
class BacktestConfig:
    season: int | None = None
    sport: str = "nfl"
    market_type: str | None = None
    require_outcomes: bool = True
    flat_stake: float = 1.0

    def resolved_market_type(self) -> str:
        if self.market_type:
            return self.market_type
        return get_sport_config(self.sport).default_market


class WalkForwardBacktester:
    """
    Walk-forward backtest: games processed in chronological kickoff order.
    Each decision uses only information available before that game's kickoff.
    """

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    def _iter_games(self, config: BacktestConfig) -> list[Game]:
        query = self.session.query(Game).filter(Game.sport == config.sport)
        if config.season is not None:
            query = query.filter(Game.season == config.season)
        if config.require_outcomes:
            query = query.filter(
                Game.home_score.isnot(None),
                Game.away_score.isnot(None),
            )
        return query.order_by(Game.kickoff_time.asc()).all()

    def _closing_line(self, game: Game, book: str, market_type: str) -> float | None:
        loader = FeatureDataLoader(self.session)
        ctx = FeatureContext.for_game(game)
        pinnacle_close = loader.latest_odds_snapshot(
            ctx, book=self.settings.pinnacle_bookmaker, market_type=market_type
        )
        if pinnacle_close and pinnacle_close.line is not None:
            return pinnacle_close.line
        soft_close = loader.latest_odds_snapshot(ctx, book=book, market_type=market_type)
        return soft_close.line if soft_close else None

    def _closing_odds(
        self, game: Game, book: str, side: str, market_type: str
    ) -> int | None:
        loader = FeatureDataLoader(self.session)
        ctx = FeatureContext.for_game(game)
        pinnacle_close = loader.latest_odds_snapshot(
            ctx, book=self.settings.pinnacle_bookmaker, market_type=market_type
        )
        if pinnacle_close is None:
            soft_close = loader.latest_odds_snapshot(ctx, book=book, market_type=market_type)
            if soft_close is None:
                return None
            return soft_close.odds_home if side == "home" else soft_close.odds_away
        return pinnacle_close.odds_home if side == "home" else pinnacle_close.odds_away

    def _settle_candidate(
        self,
        game: Game,
        candidate: BetCandidate,
        config: BacktestConfig,
        market_type: str,
    ) -> SettledBet:
        kickoff = ensure_utc(game.kickoff_time)
        side = candidate.side

        if market_type == "moneyline":
            closing_odds = self._closing_odds(
                game, candidate.book, side.value, market_type
            )
            return settle_moneyline_bet(
                game_id=game.game_id,
                book=candidate.book,
                side=side,
                american_odds=candidate.american_odds,
                closing_odds=closing_odds,
                true_prob=candidate.true_prob,
                edge_pct=candidate.edge_pct,
                stake=config.flat_stake,
                home_score=int(game.home_score),
                away_score=int(game.away_score),
                kickoff_time=kickoff,
            )

        closing_line = self._closing_line(game, candidate.book, market_type)
        if candidate.line is None:
            raise ValueError(f"Spread candidate missing line for {game.game_id}")
        return settle_spread_bet(
            game_id=game.game_id,
            book=candidate.book,
            side=side,
            american_odds=candidate.american_odds,
            line_taken=candidate.line,
            closing_line=closing_line,
            true_prob=candidate.true_prob,
            edge_pct=candidate.edge_pct,
            stake=config.flat_stake,
            home_score=int(game.home_score),
            away_score=int(game.away_score),
            kickoff_time=kickoff,
        )

    def run(
        self,
        strategy: BetStrategy | None = None,
        config: BacktestConfig | None = None,
    ):
        config = config or BacktestConfig()
        market_type = config.resolved_market_type()
        if strategy is None:
            strategy = MarketDivergenceStrategy(
                soft_book="draftkings",
                market_type=market_type,
            )

        settled: list[SettledBet] = []
        games = self._iter_games(config)

        for game in games:
            candidates: list[BetCandidate] = strategy.generate_candidates(
                self.session, game, self.settings
            )
            if not candidates:
                continue

            for candidate in candidates:
                settled.append(
                    self._settle_candidate(game, candidate, config, market_type)
                )

        return build_report(strategy_name=strategy.name, settled_bets=settled)
