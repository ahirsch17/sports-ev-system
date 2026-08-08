from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sports_ev.db.models import Base, Game


def test_schema_creates_all_tables():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    table_names = set(Base.metadata.tables.keys())
    expected = {
        "games",
        "data_sources",
        "odds_snapshots",
        "team_game_stats",
        "injury_reports",
        "model_predictions",
        "paper_bets",
        "data_quality_events",
    }
    assert expected.issubset(table_names)


def test_game_insert_and_query():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        game = Game(
            game_id="nfl_2024_w1_test",
            sport="nfl",
            home_team="HOME",
            away_team="AWAY",
            kickoff_time=datetime(2024, 9, 8, 17, 0, tzinfo=timezone.utc),
            season=2024,
            week=1,
        )
        session.add(game)
        session.commit()
        fetched = session.query(Game).filter_by(game_id="nfl_2024_w1_test").one()
        assert fetched.home_team == "HOME"
