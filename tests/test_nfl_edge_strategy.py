from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.backtest.models import BetSide
from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, OddsSnapshot, PaperBet
from sports_ev.models.nfl_edge_strategy import (
    NflEdgeMarginPredictor,
    NflEdgeSpreadStrategy,
    market_implied_margin,
    side_cover_probs,
)
from sports_ev.paper import OpportunityFlagger


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed(
    session: Session,
    *,
    kickoff: datetime,
    pinnacle_line: float | None = -3.5,
    dk_line: float = -3.5,
    dk_odds: tuple[int, int] = (-110, -110),
) -> Game:
    game = Game(
        game_id="nfl_g1",
        sport="nfl",
        home_team="Kansas City Chiefs",
        away_team="Buffalo Bills",
        kickoff_time=kickoff,
        season=2026,
    )
    session.add(game)
    captured = kickoff - timedelta(hours=2)
    if pinnacle_line is not None:
        session.add(
            OddsSnapshot(
                game_id="nfl_g1",
                book="pinnacle",
                market_type="spread",
                line=pinnacle_line,
                odds_home=-110,
                odds_away=-110,
                captured_at=captured,
                source="scraped",
            )
        )
    session.add(
        OddsSnapshot(
            game_id="nfl_g1",
            book="draftkings",
            market_type="spread",
            line=dk_line,
            odds_home=dk_odds[0],
            odds_away=dk_odds[1],
            captured_at=captured,
            source="scraped",
        )
    )
    session.commit()
    return game


class TestPricingMath:
    def test_cover_probs_symmetric_at_fair_line(self):
        assert side_cover_probs(3.5, -3.5) == pytest.approx((0.5, 0.5))
        assert side_cover_probs(3.0, -3.0) == pytest.approx((0.5, 0.5))

    def test_whole_number_line_excludes_push(self):
        p_home, p_away = side_cover_probs(5.0, -3.0)
        assert p_home + p_away == pytest.approx(1.0)
        assert p_home > 0.5

    def test_market_implied_margin_round_trips(self):
        assert market_implied_margin(-3.5, 0.5) == pytest.approx(3.5, abs=1e-6)
        mu = market_implied_margin(-7.0, 0.56)
        assert side_cover_probs(mu, -7.0)[0] == pytest.approx(0.56, abs=1e-6)


class TestStrategy:
    def test_blend_flags_home_when_model_far_above_market(self, session: Session):
        kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        game = _seed(session, kickoff=kickoff)
        strategy = NflEdgeSpreadStrategy(predictor=lambda g: 13.5, model_weight=0.5)

        blend = strategy.blended_margin(session, game, Settings())
        assert blend["market_margin"] == pytest.approx(3.5, abs=1e-6)
        assert blend["mu"] == pytest.approx(8.5, abs=1e-6)

        candidates = strategy.generate_candidates(session, game, Settings(min_edge_pct=2.5))
        assert len(candidates) == 1
        assert candidates[0].side == BetSide.HOME
        assert candidates[0].line == -3.5
        assert candidates[0].true_prob > 0.6

    def test_market_only_weight_still_finds_soft_line_gap(self, session: Session):
        kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        game = _seed(session, kickoff=kickoff, dk_line=-1.5)
        strategy = NflEdgeSpreadStrategy(predictor=lambda g: 0.0, model_weight=0.0)
        candidates = strategy.generate_candidates(session, game, Settings(min_edge_pct=2.5))
        assert [c.side for c in candidates] == [BetSide.HOME]

    def test_no_bet_when_model_agrees_with_market(self, session: Session):
        kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        game = _seed(session, kickoff=kickoff)
        strategy = NflEdgeSpreadStrategy(predictor=lambda g: 3.5, model_weight=0.15)
        assert strategy.generate_candidates(session, game, Settings(min_edge_pct=2.5)) == []

    def test_requires_sharp_line_by_default(self, session: Session):
        kickoff = datetime(2026, 9, 13, 17, 0, tzinfo=timezone.utc)
        game = _seed(session, kickoff=kickoff, pinnacle_line=None)
        strategy = NflEdgeSpreadStrategy(predictor=lambda g: 20.0, model_weight=0.5)
        assert strategy.generate_candidates(session, game, Settings()) == []

    def test_flagger_records_blend_in_audit(self, session: Session, monkeypatch):
        kickoff = datetime.now(timezone.utc) + timedelta(days=2)
        _seed(session, kickoff=kickoff, dk_odds=(150, -180))
        strategy = NflEdgeSpreadStrategy(predictor=lambda g: 3.5, model_weight=0.15)
        monkeypatch.setattr(
            "sports_ev.paper.flagging.build_nfl_edge_strategy",
            lambda **_kw: strategy,
        )

        result = OpportunityFlagger(session, Settings(min_edge_pct=2.5)).flag_upcoming(
            sport="nfl", days_ahead=7, strategy="nfl-edge"
        )
        assert result.bets_flagged == 1
        bet = session.query(PaperBet).one()
        assert bet.side == "home"
        assert bet.model_version.startswith("nfl_edge_blend_v1@w0.15")
        snap = bet.audit_trail["feature_snapshot"]
        assert snap["model_margin"] == pytest.approx(3.5)
        assert snap["market_margin"] == pytest.approx(3.5, abs=1e-6)


class TestPredictorLookup:
    def test_matches_night_game_by_eastern_date(self):
        predictor = NflEdgeMarginPredictor()
        predictor._schedules[2026] = [
            {
                "game_id": "2026_01_BUF_KC",
                "season": 2026,
                "week": 1,
                "gameday": "2026-09-10",
                "home_team": "KC",
                "away_team": "BUF",
            }
        ]
        predictor._margins[(2026, 1)] = {"2026_01_BUF_KC": 4.2}
        game = Game(
            game_id="x",
            sport="nfl",
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            kickoff_time=datetime(2026, 9, 11, 0, 20, tzinfo=timezone.utc),
            season=2026,
        )
        assert predictor(game) == pytest.approx(4.2)

    def test_unknown_matchup_returns_none(self):
        predictor = NflEdgeMarginPredictor()
        predictor._schedules[2026] = []
        game = Game(
            game_id="x",
            sport="nfl",
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            kickoff_time=datetime(2026, 9, 11, 0, 20, tzinfo=timezone.utc),
            season=2026,
        )
        assert predictor(game) is None
