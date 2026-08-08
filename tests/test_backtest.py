from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.backtest import (
    BACKTEST_VERSION,
    BacktestConfig,
    BetOutcome,
    BetSide,
    MarketDivergenceStrategy,
    WalkForwardBacktester,
    home_covered,
    settle_spread_bet,
)
from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, OddsSnapshot


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory()


def _add_game_with_odds(
    session: Session,
    *,
    game_id: str,
    kickoff: datetime,
    home_score: int,
    away_score: int,
    pinnacle_home: int,
    pinnacle_away: int,
    pinnacle_line: float,
    dk_home: int,
    dk_away: int,
    dk_line: float,
    dk_captured_offset_hours: float = -2.0,
) -> None:
    session.add(
        Game(
            game_id=game_id,
            sport="nfl",
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            kickoff_time=kickoff,
            season=2026,
            week=1,
            home_score=home_score,
            away_score=away_score,
        )
    )
    captured = kickoff + timedelta(hours=dk_captured_offset_hours)
    session.add_all(
        [
            OddsSnapshot(
                game_id=game_id,
                book="pinnacle",
                market_type="spread",
                line=pinnacle_line,
                odds_home=pinnacle_home,
                odds_away=pinnacle_away,
                captured_at=captured - timedelta(minutes=30),
                source="api",
            ),
            OddsSnapshot(
                game_id=game_id,
                book="draftkings",
                market_type="spread",
                line=dk_line,
                odds_home=dk_home,
                odds_away=dk_away,
                captured_at=captured,
                source="scraped",
            ),
        ]
    )
    session.commit()


class TestSettlement:
    def test_home_covers_negative_line(self):
        assert home_covered(home_score=24, away_score=17, home_line=-3.5) == BetOutcome.WIN

    def test_home_loses_against_spread(self):
        assert home_covered(home_score=20, away_score=17, home_line=-3.5) == BetOutcome.LOSS

    def test_push_on_exact_spread(self):
        assert home_covered(home_score=20, away_score=17, home_line=-3.0) == BetOutcome.PUSH

    def test_settle_applies_vig_on_win(self):
        bet = settle_spread_bet(
            game_id="g1",
            book="draftkings",
            side=BetSide.HOME,
            american_odds=-110,
            line_taken=-3.5,
            closing_line=-4.5,
            true_prob=0.55,
            edge_pct=5.0,
            stake=1.0,
            home_score=24,
            away_score=17,
            kickoff_time=datetime(2026, 9, 10, tzinfo=timezone.utc),
        )
        assert bet.outcome == BetOutcome.WIN
        assert bet.profit == pytest.approx(100 / 110, rel=1e-6)
        assert bet.clv == pytest.approx(1.0)

    def test_settle_loss_is_full_stake(self):
        bet = settle_spread_bet(
            game_id="g1",
            book="draftkings",
            side=BetSide.HOME,
            american_odds=-110,
            line_taken=-3.5,
            closing_line=-3.5,
            true_prob=0.55,
            edge_pct=5.0,
            stake=1.0,
            home_score=20,
            away_score=17,
            kickoff_time=datetime(2026, 9, 10, tzinfo=timezone.utc),
        )
        assert bet.outcome == BetOutcome.LOSS
        assert bet.profit == -1.0


class TestWalkForwardBacktest:
    def test_market_divergence_strategy_finds_plus_ev_bet(self, session: Session):
        kickoff = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        # Pinnacle -110/-110 => fair 50/50. DK home +150 => large edge on home.
        _add_game_with_odds(
            session,
            game_id="nfl_g1",
            kickoff=kickoff,
            home_score=24,
            away_score=17,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )

        settings = Settings(min_edge_pct=2.0)
        backtester = WalkForwardBacktester(session, settings)
        report = backtester.run(MarketDivergenceStrategy(soft_book="draftkings"))

        assert report.backtest_version == BACKTEST_VERSION
        assert report.total_bets == 1
        assert report.wins == 1
        assert report.win_rate == pytest.approx(100.0)
        assert report.roi > 0
        assert report.avg_clv == pytest.approx(0.0)
        assert report.settled_bets[0].side == BetSide.HOME
        assert report.settled_bets[0].american_odds == 150

    def test_edge_bucket_breakdown(self, session: Session):
        kickoff1 = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        kickoff2 = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)

        _add_game_with_odds(
            session,
            game_id="nfl_g1",
            kickoff=kickoff1,
            home_score=24,
            away_score=17,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )
        _add_game_with_odds(
            session,
            game_id="nfl_g2",
            kickoff=kickoff2,
            home_score=17,
            away_score=24,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )

        report = WalkForwardBacktester(session, Settings(min_edge_pct=2.0)).run()
        high_edge = next(b for b in report.edge_buckets if b.bucket == "6%+")
        assert high_edge.bets == 2
        assert high_edge.wins == 1

    def test_walk_forward_processes_games_chronologically(self, session: Session):
        kickoff_early = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        kickoff_late = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)

        _add_game_with_odds(
            session,
            game_id="nfl_late",
            kickoff=kickoff_late,
            home_score=24,
            away_score=17,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )
        _add_game_with_odds(
            session,
            game_id="nfl_early",
            kickoff=kickoff_early,
            home_score=24,
            away_score=17,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )

        report = WalkForwardBacktester(session, Settings(min_edge_pct=2.0)).run()
        kickoffs = [b.kickoff_time for b in report.settled_bets]
        assert kickoffs == sorted(kickoffs)

    def test_reproducibility_hash_is_stable(self, session: Session):
        kickoff = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        _add_game_with_odds(
            session,
            game_id="nfl_g1",
            kickoff=kickoff,
            home_score=24,
            away_score=17,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )

        backtester = WalkForwardBacktester(session, Settings(min_edge_pct=2.0))
        report_a = backtester.run()
        report_b = backtester.run()
        assert report_a.reproducibility_hash() == report_b.reproducibility_hash()

    def test_max_drawdown_computed(self, session: Session):
        kickoff1 = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        kickoff2 = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)

        _add_game_with_odds(
            session,
            game_id="win_game",
            kickoff=kickoff1,
            home_score=24,
            away_score=17,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )
        _add_game_with_odds(
            session,
            game_id="loss_game",
            kickoff=kickoff2,
            home_score=17,
            away_score=24,
            pinnacle_home=-110,
            pinnacle_away=-110,
            pinnacle_line=-3.5,
            dk_home=150,
            dk_away=-180,
            dk_line=-3.5,
        )

        report = WalkForwardBacktester(session, Settings(min_edge_pct=2.0)).run()
        assert report.max_drawdown > 0

    def test_skips_games_without_outcomes(self, session: Session):
        kickoff = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        session.add(
            Game(
                game_id="pending",
                sport="nfl",
                home_team="Kansas City Chiefs",
                away_team="Buffalo Bills",
                kickoff_time=kickoff,
                season=2026,
                week=1,
            )
        )
        session.commit()

        report = WalkForwardBacktester(session).run()
        assert report.total_bets == 0
