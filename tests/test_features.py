from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, InjuryReport, OddsSnapshot, TeamGameStat
from sports_ev.features import FeaturePipeline, LeakageError
from sports_ev.features.leakage import FeatureAudit, assert_no_leakage
from sports_ev.features.time_utils import ensure_utc
from sports_ev.features.version import feature_version
from sports_ev.pricing import DevigMethod


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sess = factory()

    kickoff_w2 = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)
    kickoff_w1 = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)

    sess.add_all(
        [
            Game(
                game_id="nfl_w1",
                sport="nfl",
                home_team="Kansas City Chiefs",
                away_team="Baltimore Ravens",
                kickoff_time=kickoff_w1,
                season=2026,
                week=1,
            ),
            Game(
                game_id="nfl_w2",
                sport="nfl",
                home_team="Kansas City Chiefs",
                away_team="Buffalo Bills",
                kickoff_time=kickoff_w2,
                season=2026,
                week=2,
            ),
        ]
    )

    # Week 1 stats — knowable before week 2 kickoff
    sess.add_all(
        [
            TeamGameStat(
                game_id="nfl_w1",
                team="Kansas City Chiefs",
                stat_key="epa_offense",
                stat_value=0.12,
                known_at=kickoff_w1 + timedelta(hours=3),
            ),
            TeamGameStat(
                game_id="nfl_w1",
                team="Kansas City Chiefs",
                stat_key="epa_defense",
                stat_value=-0.05,
                known_at=kickoff_w1 + timedelta(hours=3),
            ),
            TeamGameStat(
                game_id="nfl_w1",
                team="Kansas City Chiefs",
                stat_key="wr_target_share",
                stat_value=0.28,
                known_at=kickoff_w1 + timedelta(hours=3),
            ),
            TeamGameStat(
                game_id="nfl_w1",
                team="Kansas City Chiefs",
                stat_key="wr_epa",
                stat_value=0.20,
                known_at=kickoff_w1 + timedelta(hours=3),
            ),
            TeamGameStat(
                game_id="nfl_w1",
                team="Buffalo Bills",
                stat_key="pass_def_epa",
                stat_value=-0.08,
                known_at=kickoff_w1 + timedelta(hours=3),
            ),
            TeamGameStat(
                game_id="nfl_w1",
                team="Buffalo Bills",
                stat_key="epa_offense",
                stat_value=0.08,
                known_at=kickoff_w1 + timedelta(hours=3),
            ),
        ]
    )

    # Pinnacle + soft book odds before week 2 kickoff
    sess.add_all(
        [
            OddsSnapshot(
                game_id="nfl_w2",
                book="pinnacle",
                market_type="spread",
                captured_at=kickoff_w2 - timedelta(hours=2),
                odds_home=-110,
                odds_away=-110,
                line=-3.5,
                source="api",
            ),
            OddsSnapshot(
                game_id="nfl_w2",
                book="draftkings",
                market_type="spread",
                captured_at=kickoff_w2 - timedelta(hours=6),
                odds_home=-105,
                odds_away=-115,
                line=-2.5,
                source="scraped",
            ),
            OddsSnapshot(
                game_id="nfl_w2",
                book="draftkings",
                market_type="spread",
                captured_at=kickoff_w2 - timedelta(hours=1),
                odds_home=-110,
                odds_away=-110,
                line=-3.5,
                source="scraped",
            ),
        ]
    )

    sess.add(
        InjuryReport(
            game_id="nfl_w2",
            team="Kansas City Chiefs",
            player="Patrick Mahomes",
            status="questionable",
            report_timestamp=kickoff_w2 - timedelta(hours=5),
        )
    )

    sess.commit()
    yield sess
    sess.close()


class TestLeakageGuard:
    def test_assert_no_leakage_passes_before_kickoff(self):
        kickoff = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)
        audit = FeatureAudit(kickoff_time=kickoff)
        audit.add("test", 1.0, kickoff - timedelta(hours=1), "test")
        assert_no_leakage(audit)

    def test_assert_no_leakage_fails_at_kickoff(self):
        kickoff = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)
        audit = FeatureAudit(kickoff_time=kickoff)
        audit.add("test", 1.0, kickoff, "test")
        with pytest.raises(LeakageError):
            assert_no_leakage(audit)

    def test_pipeline_rejects_leaky_audit_entry(self, session: Session):
        """Simulates a feature builder that records a post-kickoff source timestamp."""
        from sports_ev.features.leakage import FeatureAudit, assert_no_leakage

        kickoff = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)
        audit = FeatureAudit(kickoff_time=kickoff)
        audit.add("bad_feature", 1.0, kickoff + timedelta(seconds=1), "simulated")
        with pytest.raises(LeakageError):
            assert_no_leakage(audit)

    def test_loader_excludes_post_kickoff_stats(self, session: Session):
        from sports_ev.features.sources import FeatureContext, FeatureDataLoader

        game = session.query(Game).filter_by(game_id="nfl_w2").one()
        ctx = FeatureContext.for_game(game)
        session.add(
            TeamGameStat(
                game_id="nfl_w1",
                team="Kansas City Chiefs",
                stat_key="epa_offense",
                stat_value=0.99,
                known_at=ctx.kickoff_time + timedelta(hours=1),
            )
        )
        session.commit()
        history = FeatureDataLoader(session).prior_team_stats(ctx, "Kansas City Chiefs")
        assert all(ensure_utc(s.known_at) < ctx.kickoff_time for s in history.stats)
        assert not any(s.stat_value == 0.99 for s in history.stats)

    def test_pipeline_rejects_post_kickoff_odds(self, session: Session):
        kickoff = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)
        session.add(
            OddsSnapshot(
                game_id="nfl_w2",
                book="pinnacle",
                market_type="spread",
                captured_at=kickoff + timedelta(seconds=1),
                odds_home=-110,
                odds_away=-110,
                line=-3.5,
                source="api",
            )
        )
        session.commit()

        pipeline = FeaturePipeline(session)
        # Latest pre-kickoff pinnacle still exists; post-kickoff shouldn't be picked
        # but if we only had post-kickoff, we'd get no pinnacle features - test leakage on forced path
        session.query(OddsSnapshot).filter(OddsSnapshot.captured_at < kickoff).delete()
        session.commit()
        # With only post-kickoff odds, loader filters them out — no leakage, but also no market features
        vector = pipeline.build_for_game("nfl_w2")
        assert vector.features.get("pinnacle_fair_prob_home") is None


class TestFeaturePipeline:
    def test_builds_versioned_features(self, session: Session):
        pipeline = FeaturePipeline(session, Settings(default_devig_method="multiplicative"))
        vector = pipeline.build_for_game("nfl_w2")

        assert vector.feature_version == feature_version()
        assert vector.game_id == "nfl_w2"
        assert vector.features["home_epa_offense_season"] == pytest.approx(0.12)
        assert vector.features["pinnacle_fair_prob_home"] == pytest.approx(0.5, abs=0.01)
        assert vector.features["draftkings_line_move"] == pytest.approx(-1.0)
        assert vector.features["home_injury_status_changed_24h"] == 1.0
        assert vector.features["home_rest_days"] == pytest.approx(7.0, abs=0.01)
        assert len(vector.audit_sources) > 0

    def test_shin_devig_toggle(self, session: Session):
        pipeline = FeaturePipeline(session)
        mult = pipeline.build_for_game("nfl_w2", devig_method=DevigMethod.MULTIPLICATIVE)
        shin = pipeline.build_for_game("nfl_w2", devig_method=DevigMethod.SHIN)
        assert mult.features["pinnacle_fair_prob_home"] == pytest.approx(
            shin.features["pinnacle_fair_prob_home"], abs=1e-9
        )

    def test_soft_book_divergence(self, session: Session):
        pipeline = FeaturePipeline(session)
        vector = pipeline.build_for_game("nfl_w2")
        div = vector.features.get("draftkings_pinnacle_divergence")
        assert div is not None
        assert isinstance(div, float)

    def test_matchup_feature(self, session: Session):
        pipeline = FeaturePipeline(session)
        vector = pipeline.build_for_game("nfl_w2")
        matchup = vector.features.get("matchup_home_wr_vs_away_pass_def")
        assert matchup is not None
        # 0.20 * 0.28 - (-0.08) = 0.136
        assert matchup == pytest.approx(0.136, abs=0.001)
