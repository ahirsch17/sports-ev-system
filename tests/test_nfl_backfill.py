import csv
import io
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sports_ev.db.models import Base, Game, OddsSnapshot
from sports_ev.stats.nfl_backfill import NflHistoryBackfillService, _home_spread_line


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sess = factory()
    yield sess
    sess.close()


def test_home_spread_line_convention():
    assert _home_spread_line("3") == -3.0
    assert _home_spread_line("-3") == -3.0


def test_backfill_seasons_upserts_game(session):
    games_csv = (
        "game_id,season,game_type,week,gameday,gametime,away_team,away_score,home_team,home_score,spread_line\n"
        "2024_01_BAL_KC,2024,REG,1,2024-09-05,20:20,BAL,20,KC,27,3\n"
    )
    teams_csv = "team_abbr,team_name\nKC,Kansas City Chiefs\nBAL,Baltimore Ravens\n"
    week_csv = (
        "season,week,team,game_id,passing_epa,rushing_epa,success_rate\n"
        "2024,1,KC,2024_01_BAL_KC,5.0,2.0,0.45\n"
    )

    def fake_fetch(url: str) -> str:
        if "games.csv" in url:
            return games_csv
        if "teams_colors_logos.csv" in url:
            return teams_csv
        if "stats_team_week" in url:
            return week_csv
        raise AssertionError(url)

    with patch("sports_ev.stats.nfl_backfill._fetch_text", side_effect=fake_fetch):
        result = NflHistoryBackfillService(session).backfill_seasons([2024])

    assert result.games_upserted == 1
    game = session.query(Game).filter_by(game_id="nfl_2024_01_BAL_KC").one()
    assert game.home_team == "Kansas City Chiefs"
    assert game.home_score == 27
    odds = session.query(OddsSnapshot).filter_by(game_id=game.game_id).all()
    assert len(odds) == 2
    assert odds[0].line == -3.0
