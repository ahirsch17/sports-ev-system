from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, PaperBet
from sports_ev.paper.report import build_paper_report


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory()


def _add_bet(session: Session, *, game_id: str, edge: float, outcome: str | None = "win") -> None:
    kickoff = datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)
    flagged = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    session.add(
        Game(
            game_id=game_id,
            sport="mlb",
            home_team="Braves",
            away_team="Padres",
            kickoff_time=kickoff,
            season=2026,
            home_score=5,
            away_score=3,
        )
    )
    session.add(
        PaperBet(
            game_id=game_id,
            book="draftkings",
            market_type="moneyline",
            side="home",
            flagged_at=flagged,
            suggested_stake_pct=1.0,
            odds_taken=-110,
            line_taken=None,
            model_version="test",
            edge_pct=edge,
            audit_trail={"test": True},
            outcome=outcome,
            clv=1.5,
            settled_at=flagged if outcome else None,
        )
    )
    session.commit()


def test_paper_report_includes_bankroll_and_plus_ev(session: Session):
    _add_bet(session, game_id="g1", edge=3.0, outcome="win")
    _add_bet(session, game_id="g2", edge=1.0, outcome="loss")
    report = build_paper_report(session, days=30, settings=Settings(min_edge_pct=2.5))
    assert "Bankroll" in report.text
    assert "+EV only" in report.text
    assert "3.0" in report.text or "+3.0" in report.text
