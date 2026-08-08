from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, OddsSnapshot, TeamGameStat
from sports_ev.features.mlb.pipeline import MlbFeaturePipeline
from sports_ev.features.version import feature_version


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sess = factory()

    game1 = datetime(2026, 7, 20, 23, 0, tzinfo=timezone.utc)
    game2 = datetime(2026, 7, 23, 23, 0, tzinfo=timezone.utc)

    sess.add_all(
        [
            Game(
                game_id="mlb_g1",
                sport="mlb",
                home_team="New York Yankees",
                away_team="Boston Red Sox",
                kickoff_time=game1,
                season=2026,
                home_score=5,
                away_score=3,
            ),
            Game(
                game_id="mlb_g2",
                sport="mlb",
                home_team="New York Yankees",
                away_team="Tampa Bay Rays",
                kickoff_time=game2,
                season=2026,
            ),
        ]
    )

    known = game1 + timedelta(hours=4)
    for team, rs, ra in [
        ("New York Yankees", 5, 3),
        ("Boston Red Sox", 3, 5),
    ]:
        for key, val in [
            ("runs_scored", rs),
            ("runs_allowed", ra),
            ("starter_innings", 6.0),
            ("starter_earned_runs", 2.0),
            ("bullpen_earned_runs", 1.0),
            ("hits", 8.0),
        ]:
            sess.add(
                TeamGameStat(
                    game_id="mlb_g1",
                    team=team,
                    stat_key=key,
                    stat_value=float(val),
                    known_at=known,
                )
            )

    sess.add(
        OddsSnapshot(
            game_id="mlb_g2",
            book="pinnacle",
            market_type="moneyline",
            captured_at=game2 - timedelta(hours=2),
            odds_home=-150,
            odds_away=130,
            source="scraped",
        )
    )
    sess.add_all(
        [
            TeamGameStat(
                game_id="mlb_g2",
                team="New York Yankees",
                stat_key="probable_starter_era",
                stat_value=3.25,
                known_at=game2 - timedelta(hours=3),
            ),
            TeamGameStat(
                game_id="mlb_g2",
                team="Tampa Bay Rays",
                stat_key="probable_starter_era",
                stat_value=4.10,
                known_at=game2 - timedelta(hours=3),
            ),
        ]
    )
    sess.commit()
    yield sess
    sess.close()


class TestMlbFeaturePipeline:
    def test_builds_versioned_features(self, session: Session):
        pipeline = MlbFeaturePipeline(session, Settings(default_devig_method="multiplicative"))
        vector = pipeline.build_for_game("mlb_g2")

        assert vector.feature_version == feature_version("mlb")
        assert vector.features["home_runs_scored_season"] == pytest.approx(5.0)
        assert vector.features["home_park_factor"] == pytest.approx(1.04)
        assert vector.features["home_probable_era"] == pytest.approx(3.25)
        assert vector.features["matchup_probable_era_delta"] == pytest.approx(4.10 - 3.25)

    def test_leakage_guard_excludes_future_stats(self, session: Session):
        game2 = session.query(Game).filter_by(game_id="mlb_g2").one()
        session.add(
            TeamGameStat(
                game_id="mlb_g1",
                team="New York Yankees",
                stat_key="runs_scored",
                stat_value=99.0,
                known_at=game2.kickoff_time + timedelta(hours=1),
            )
        )
        session.commit()

        pipeline = MlbFeaturePipeline(session)
        vector = pipeline.build_for_game("mlb_g2")
        assert vector.features["home_runs_scored_season"] == pytest.approx(5.0)
