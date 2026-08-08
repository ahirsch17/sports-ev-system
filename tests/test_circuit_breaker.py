from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings
from sports_ev.db.models import Base, DataQualityEvent, DataSource, Game, OddsSnapshot, SourceStatus
from sports_ev.integrity.circuit_breaker import CircuitBreaker
from sports_ev.integrity.validators import OddsSnapshotPayload


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        circuit_breaker_failure_threshold=3,
        circuit_breaker_max_line_jump=15.0,
        circuit_breaker_max_stale_seconds=3600,
    )


@pytest.fixture
def session(settings: Settings) -> Session:
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sess = factory()
    # Seed a game for FK constraints
    sess.add(
        Game(
            game_id="nfl_2024_w1_kc_bal",
            sport="nfl",
            home_team="KC",
            away_team="BAL",
            kickoff_time=datetime(2024, 9, 5, 20, 20, tzinfo=timezone.utc),
            season=2024,
            week=1,
        )
    )
    sess.commit()
    yield sess
    sess.close()


def _valid_payload(**overrides) -> OddsSnapshotPayload:
    base = dict(
        game_id="nfl_2024_w1_kc_bal",
        book="draftkings",
        market_type="spread",
        captured_at=datetime.now(timezone.utc),
        odds_home=-110,
        odds_away=-110,
        line=-3.5,
        source="scraped",
        parser_version="json_api@1.0",
    )
    base.update(overrides)
    return OddsSnapshotPayload(**base)


class TestCircuitBreaker:
    def test_valid_snapshot_accepted(self, session: Session, settings: Settings):
        cb = CircuitBreaker(session, settings)
        result = cb.ingest_odds_snapshot(_valid_payload())

        assert result.accepted is True
        assert result.suppressed is False
        assert result.snapshot_id is not None
        assert session.query(OddsSnapshot).count() == 1

    def test_invalid_odds_rejected(self, session: Session, settings: Settings):
        cb = CircuitBreaker(session, settings)
        result = cb.ingest_odds_snapshot(_valid_payload(odds_home=999999))

        assert result.accepted is False
        assert session.query(OddsSnapshot).count() == 0
        assert session.query(DataQualityEvent).count() >= 1

    def test_trips_after_n_failures(self, session: Session, settings: Settings):
        cb = CircuitBreaker(session, settings)

        for _ in range(3):
            cb.ingest_odds_snapshot(_valid_payload(odds_home=999999))

        source = session.query(DataSource).filter_by(source_key="draftkings:spread").one()
        assert source.status == SourceStatus.DEGRADED
        assert source.consecutive_failures == 3

        events = (
            session.query(DataQualityEvent)
            .filter_by(event_type="circuit_breaker_tripped")
            .all()
        )
        assert len(events) == 1

    def test_degraded_source_suppresses_even_valid_data(
        self, session: Session, settings: Settings
    ):
        cb = CircuitBreaker(session, settings)

        for _ in range(3):
            cb.ingest_odds_snapshot(_valid_payload(line=999.0))

        result = cb.ingest_odds_snapshot(_valid_payload())
        assert result.accepted is False
        assert result.suppressed is True
        assert session.query(OddsSnapshot).count() == 0

    def test_clear_degraded_allows_ingestion(self, session: Session, settings: Settings):
        cb = CircuitBreaker(session, settings)

        for _ in range(3):
            cb.ingest_odds_snapshot(_valid_payload(odds_home=0))

        cb.clear_degraded("draftkings:spread", reviewed_by="test")
        result = cb.ingest_odds_snapshot(_valid_payload())

        assert result.accepted is True
        source = session.query(DataSource).filter_by(source_key="draftkings:spread").one()
        assert source.status == SourceStatus.HEALTHY

    def test_line_jump_rejected(self, session: Session, settings: Settings):
        cb = CircuitBreaker(session, settings)
        cb.ingest_odds_snapshot(_valid_payload(line=-3.5))

        result = cb.ingest_odds_snapshot(_valid_payload(line=20.0))
        assert result.accepted is False
        assert any("Line jump" in f.message for f in result.failures)

    def test_stale_data_rejected(self, session: Session, settings: Settings):
        cb = CircuitBreaker(session, settings)
        stale_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = cb.ingest_odds_snapshot(_valid_payload(captured_at=stale_time))

        assert result.accepted is False
        assert any(f.reason == "freshness" for f in result.failures)
