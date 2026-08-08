from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import get_settings
from sports_ev.db.models import Base, Game, OddsSnapshot
from sports_ev.odds.closing_capture import ClosingLineCaptureService
from sports_ev.paper.settlement import _closing_odds


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory()


def _seed_started_game(session: Session, *, game_id: str = "g1") -> Game:
    kickoff = datetime.now(timezone.utc) - timedelta(hours=2)
    game = Game(
        game_id=game_id,
        sport="mlb",
        home_team="Braves",
        away_team="Padres",
        kickoff_time=kickoff,
        season=2026,
    )
    session.add(game)
    session.add(
        OddsSnapshot(
            game_id=game_id,
            book="pinnacle",
            market_type="moneyline",
            captured_at=kickoff - timedelta(minutes=15),
            odds_home=-150,
            odds_away=130,
            source="scraped",
        )
    )
    session.commit()
    return game


class TestClosingLineCapture:
    def test_captures_pinnacle_closing_before_kickoff(self, session: Session):
        game = _seed_started_game(session)
        result = ClosingLineCaptureService(session).capture_recent(lookback_hours=24)
        session.commit()

        assert result.games_captured == 1
        refreshed = session.query(Game).filter_by(game_id=game.game_id).one()
        assert refreshed.pinnacle_closing_home_odds == -150
        assert refreshed.pinnacle_closing_away_odds == 130
        assert refreshed.closing_captured_at is not None

    def test_skips_already_captured(self, session: Session):
        game = _seed_started_game(session, game_id="g2")
        service = ClosingLineCaptureService(session)
        service.capture_recent(lookback_hours=24)
        session.commit()
        result = service.capture_recent(lookback_hours=24)
        assert result.skipped_existing == 1
        assert result.games_captured == 0

    def test_settlement_uses_stored_closing_odds(self, session: Session):
        game = _seed_started_game(session, game_id="g3")
        ClosingLineCaptureService(session).capture_recent(lookback_hours=24)
        session.commit()
        game = session.query(Game).filter_by(game_id="g3").one()
        closing = _closing_odds(
            session,
            game,
            settings=get_settings(),
            book="draftkings",
            side="home",
            market_type="moneyline",
        )
        assert closing == -150
