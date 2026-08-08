from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sports_ev.db.models import Base, Game, OddsSnapshot, TeamGameStat
from sports_ev.models.mlb_dataset import build_mlb_labeled_rows
from sports_ev.models.mlb_train import train_mlb_from_rows


def _seed_mlb_season(session, *, count: int = 24) -> None:
    base = datetime(2026, 4, 1, 18, 0, tzinfo=timezone.utc)
    teams = [
        ("New York Yankees", "Boston Red Sox"),
        ("Los Angeles Dodgers", "San Francisco Giants"),
    ]
    for i in range(count):
        home, away = teams[i % 2]
        kickoff = base + timedelta(days=i)
        game_id = f"mlb_train_{i}"
        home_score = 4 + (i % 4)
        away_score = 2 + (i % 5)
        if home_score == away_score:
            home_score += 1
        session.add(
            Game(
                game_id=game_id,
                sport="mlb",
                home_team=home,
                away_team=away,
                kickoff_time=kickoff,
                season=2026,
                home_score=home_score,
                away_score=away_score,
            )
        )
        known = kickoff + timedelta(hours=4)
        for team, rs, ra in [(home, home_score, away_score), (away, away_score, home_score)]:
            for key, val in [
                ("runs_scored", rs),
                ("runs_allowed", ra),
                ("starter_innings", 6.0),
                ("starter_earned_runs", 2.0),
                ("bullpen_earned_runs", 1.0),
                ("hits", 7.0),
            ]:
                session.add(
                    TeamGameStat(
                        game_id=game_id,
                        team=team,
                        stat_key=key,
                        stat_value=float(val),
                        known_at=known,
                    )
                )
        session.add(
            OddsSnapshot(
                game_id=game_id,
                book="pinnacle",
                market_type="moneyline",
                captured_at=kickoff - timedelta(hours=1),
                odds_home=-130 if home_score >= away_score else 110,
                odds_away=110 if home_score >= away_score else -130,
                source="scraped",
            )
        )


class TestMlbModel:
    def test_train_and_predict(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        _seed_mlb_season(session, count=40)
        session.commit()

        rows = build_mlb_labeled_rows(session)
        assert len(rows) >= 20

        model, metrics = train_mlb_from_rows(rows)
        assert metrics.train_rows >= 15
        assert model.feature_names

        pred = model.predict_one(rows[-1].features, compute_shap=False)
        assert 0.0 <= pred.home_win_prob <= 1.0
        assert pred.away_win_prob == pytest.approx(1.0 - pred.home_win_prob)
        assert 0.10 <= pred.home_win_prob <= 0.90
        assert model.feature_imputations
