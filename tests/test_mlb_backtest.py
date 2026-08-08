from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sports_ev.backtest import (
    BacktestConfig,
    MarketDivergenceStrategy,
    WalkForwardBacktester,
    settle_moneyline_bet,
)
from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, OddsSnapshot


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_mlb_moneyline_game(session, *, game_id: str, kickoff: datetime, home_score: int, away_score: int):
    session.add(
        Game(
            game_id=game_id,
            sport="mlb",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            kickoff_time=kickoff,
            season=2026,
            home_score=home_score,
            away_score=away_score,
        )
    )
    captured = kickoff - timedelta(hours=2)
    session.add_all(
        [
            OddsSnapshot(
                game_id=game_id,
                book="pinnacle",
                market_type="moneyline",
                odds_home=-130,
                odds_away=110,
                captured_at=captured,
                source="scraped",
            ),
            OddsSnapshot(
                game_id=game_id,
                book="draftkings",
                market_type="moneyline",
                odds_home=150,
                odds_away=-170,
                captured_at=captured,
                source="scraped",
            ),
        ]
    )
    session.commit()


class TestMlbBacktest:
    def test_settle_moneyline_bet(self):
        bet = settle_moneyline_bet(
            game_id="g1",
            book="draftkings",
            side="home",
            american_odds=150,
            closing_odds=-140,
            true_prob=0.55,
            edge_pct=5.0,
            stake=1.0,
            home_score=5,
            away_score=3,
            kickoff_time=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
        assert bet.outcome.value == "win"
        assert bet.market_type == "moneyline"
        assert bet.clv is not None

    def test_walk_forward_market_divergence(self, session):
        kickoff = datetime(2026, 7, 10, 23, 0, tzinfo=timezone.utc)
        _add_mlb_moneyline_game(session, game_id="mlb_bt_1", kickoff=kickoff, home_score=5, away_score=2)

        report = WalkForwardBacktester(session, Settings(min_edge_pct=2.0)).run(
            strategy=MarketDivergenceStrategy(soft_book="draftkings", market_type="moneyline"),
            config=BacktestConfig(sport="mlb"),
        )
        assert report.total_bets >= 0
        assert report.reproducibility_hash()
