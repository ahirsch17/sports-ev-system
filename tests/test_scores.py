from __future__ import annotations

import pytest

from sports_ev.scores.espn import parse_espn_scoreboard
from sports_ev.scores.ingest import ScoreIngestService
from sports_ev.softbooks.pinnacle_parser import parse_pinnacle_moneylines


ESPN_PAYLOAD = {
    "events": [
        {
            "date": "2026-07-22T23:05:00Z",
            "competitions": [
                {
                    "status": {"type": {"completed": True, "state": "post"}},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "5",
                            "team": {"displayName": "New York Yankees"},
                        },
                        {
                            "homeAway": "away",
                            "score": "3",
                            "team": {"displayName": "Boston Red Sox"},
                        },
                    ],
                }
            ],
        }
    ]
}

MLB_MATCHUP = {
    "id": 999001,
    "type": "matchup",
    "isLive": False,
    "startTime": "2026-07-23T23:05:00Z",
    "participants": [
        {"alignment": "away", "name": "Boston Red Sox"},
        {"alignment": "home", "name": "New York Yankees"},
    ],
}

MLB_MONEYLINE_MARKET = {
    "matchupId": 999001,
    "type": "moneyline",
    "period": 0,
    "isAlternate": False,
    "prices": [
        {"designation": "home", "price": -150},
        {"designation": "away", "price": 130},
    ],
}


class TestEspnScores:
    def test_parse_completed_game(self):
        scores = parse_espn_scoreboard(ESPN_PAYLOAD)
        assert len(scores) == 1
        score = scores[0]
        assert score.home_team == "New York Yankees"
        assert score.away_team == "Boston Red Sox"
        assert score.home_score == 5
        assert score.away_score == 3
        assert score.completed is True


class TestPinnacleMoneylineParser:
    def test_parse_moneyline(self):
        lines = parse_pinnacle_moneylines([MLB_MATCHUP], [MLB_MONEYLINE_MARKET])
        assert len(lines) == 1
        line = lines[0]
        assert line.home_team == "New York Yankees"
        assert line.odds_home == -150
        assert line.odds_away == 130


class TestScoreIngestService:
    @pytest.fixture
    def session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from sports_ev.db.models import Base, Game

        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        sess = factory()
        from datetime import datetime, timezone

        sess.add(
            Game(
                game_id="mlb_pin_999001",
                sport="mlb",
                home_team="New York Yankees",
                away_team="Boston Red Sox",
                kickoff_time=datetime(2026, 7, 22, 23, 5, tzinfo=timezone.utc),
                season=2026,
            )
        )
        sess.commit()
        return sess

    def test_ingest_updates_game_scores(self, session, monkeypatch):
        from datetime import date

        class FakeClient:
            def fetch_completed_scores(self, path, *, on_date=None):
                return parse_espn_scoreboard(ESPN_PAYLOAD)

            def close(self):
                pass

        monkeypatch.setattr(
            "sports_ev.scores.ingest.date",
            type("D", (), {"today": staticmethod(lambda: date(2026, 7, 22))}),
        )
        service = ScoreIngestService(session, client=FakeClient())
        result = service.ingest_sport("mlb", lookback_days=0)
        assert result.scores_seen == 1
        assert result.games_updated == 1

        from sports_ev.db.models import Game

        game = session.query(Game).filter_by(game_id="mlb_pin_999001").one()
        assert game.home_score == 5
        assert game.away_score == 3
