import pytest
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.db.migrate import migrate_db


@pytest.fixture
def legacy_engine():
    """Simulate a pre-Phase-2 games table missing closing-line columns."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()
    Table(
        "games",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("game_id", String, nullable=False),
        Column("sport", String, nullable=False),
        Column("home_team", String, nullable=False),
        Column("away_team", String, nullable=False),
        Column("kickoff_time", DateTime),
        Column("home_score", Integer),
        Column("away_score", Integer),
    )
    metadata.create_all(engine)
    return engine


def test_migrate_db_adds_closing_columns(legacy_engine):
    applied = migrate_db(engine=legacy_engine)
    assert "games.pinnacle_closing_spread" in applied
    assert len(applied) == 4

    applied_again = migrate_db(engine=legacy_engine)
    assert applied_again == []

    cols = {col["name"] for col in inspect(legacy_engine).get_columns("games")}
    assert "pinnacle_closing_spread" in cols
    assert "pinnacle_closing_home_odds" in cols
    assert "pinnacle_closing_away_odds" in cols
    assert "closing_captured_at" in cols
