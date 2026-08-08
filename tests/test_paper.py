from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings
from sports_ev.db.models import Base, DataSource, Game, OddsSnapshot, PaperBet, SourceStatus
from sports_ev.paper import (
    ManualBetPlacer,
    ManualBetRequest,
    OpportunityFlagger,
    PaperBetSettlementService,
    clv_trend,
    compute_summary,
    kelly_fraction,
)
from sports_ev.paper.kelly import expected_profit_pct


class TestKelly:
    def test_positive_edge_yields_stake(self):
        stake = kelly_fraction(0.55, -110, fraction=0.25, max_stake_pct=5.0)
        assert stake > 0
        assert stake <= 5.0

    def test_no_edge_yields_zero(self):
        assert kelly_fraction(0.45, -110, fraction=0.25) == 0.0

    def test_respects_max_cap(self):
        stake = kelly_fraction(0.70, 200, fraction=1.0, max_stake_pct=2.0)
        assert stake == pytest.approx(2.0)

    def test_expected_profit_pct_win(self):
        profit = expected_profit_pct(2.0, -110, won=True)
        assert profit == pytest.approx(2.0 * (100 / 110))

    def test_expected_profit_pct_loss(self):
        assert expected_profit_pct(2.0, -110, won=False) == -2.0


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory()


def _seed_upcoming_game(session: Session, *, game_id: str, kickoff: datetime) -> None:
    session.add(
        Game(
            game_id=game_id,
            sport="nfl",
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            kickoff_time=kickoff,
            season=2026,
            week=1,
        )
    )
    captured = kickoff - timedelta(hours=2)
    session.add_all(
        [
            OddsSnapshot(
                game_id=game_id,
                book="pinnacle",
                market_type="spread",
                line=-3.5,
                odds_home=-110,
                odds_away=-110,
                captured_at=captured - timedelta(minutes=30),
                source="api",
            ),
            OddsSnapshot(
                game_id=game_id,
                book="draftkings",
                market_type="spread",
                line=-3.5,
                odds_home=150,
                odds_away=-180,
                captured_at=captured,
                source="scraped",
            ),
        ]
    )


def _seed_settled_game(session: Session, *, game_id: str, kickoff: datetime) -> None:
    _seed_upcoming_game(session, game_id=game_id, kickoff=kickoff)
    game = session.query(Game).filter_by(game_id=game_id).one()
    game.home_score = 24
    game.away_score = 17


class TestFlagging:
    def test_flags_plus_ev_opportunity(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=2)
        _seed_upcoming_game(session, game_id="upcoming_1", kickoff=kickoff)
        session.commit()

        result = OpportunityFlagger(session, Settings(min_edge_pct=2.0)).flag_upcoming(
            days_ahead=7, strategy="market", soft_book="draftkings"
        )
        assert result.games_scanned == 1
        assert result.bets_flagged == 1

        bet = session.query(PaperBet).one()
        assert bet.outcome == "pending"
        assert bet.edge_pct > 0
        assert bet.suggested_stake_pct > 0
        assert "strategy" in bet.audit_trail
        assert "feature_snapshot" in bet.audit_trail
        assert bet.audit_trail["american_odds"] == 150

    def test_skips_duplicate_pending_bet(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=2)
        _seed_upcoming_game(session, game_id="upcoming_1", kickoff=kickoff)
        session.commit()

        flagger = OpportunityFlagger(session, Settings(min_edge_pct=2.0))
        first = flagger.flag_upcoming(days_ahead=7, strategy="market")
        second = flagger.flag_upcoming(days_ahead=7, strategy="market")
        session.commit()

        assert first.bets_flagged == 1
        assert second.skipped_existing == 1
        assert session.query(PaperBet).count() == 1

    def test_suppresses_when_source_degraded(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=2)
        _seed_upcoming_game(session, game_id="upcoming_1", kickoff=kickoff)
        session.add(
            DataSource(
                source_key="draftkings:spread",
                source_type="scraped",
                status=SourceStatus.DEGRADED,
                consecutive_failures=3,
            )
        )
        session.commit()

        result = OpportunityFlagger(session).flag_upcoming(days_ahead=7)
        assert result.bets_flagged == 0
        assert result.suppressed_degraded == 1


class TestSettlement:
    def test_settles_pending_bet_with_clv(self, session: Session):
        kickoff = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        _seed_settled_game(session, game_id="g1", kickoff=kickoff)
        session.add(
            PaperBet(
                game_id="g1",
                book="draftkings",
                market_type="spread",
                side="home",
                flagged_at=kickoff - timedelta(hours=3),
                suggested_stake_pct=1.5,
                odds_taken=-110,
                line_taken=-3.5,
                model_version="market_divergence_v1",
                edge_pct=5.0,
                audit_trail={"strategy": "market_divergence_v1"},
                outcome="pending",
            )
        )
        session.commit()

        result = PaperBetSettlementService(session).settle_pending()
        session.commit()

        assert result.settled == 1
        bet = session.query(PaperBet).one()
        assert bet.outcome == "win"
        assert bet.clv == pytest.approx(0.0)
        assert bet.settled_at is not None

    def test_leaves_unsettled_games_pending(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=3)
        _seed_upcoming_game(session, game_id="g1", kickoff=kickoff)
        session.add(
            PaperBet(
                game_id="g1",
                book="draftkings",
                market_type="spread",
                side="home",
                flagged_at=kickoff - timedelta(hours=1),
                suggested_stake_pct=1.0,
                odds_taken=-110,
                line_taken=-3.5,
                model_version="market_divergence_v1",
                edge_pct=3.0,
                audit_trail={},
                outcome="pending",
            )
        )
        session.commit()

        result = PaperBetSettlementService(session).settle_pending()
        assert result.still_pending == 1
        assert result.settled == 0

    def test_settles_moneyline_bet(self, session: Session):
        kickoff = datetime(2026, 7, 22, 23, 0, tzinfo=timezone.utc)
        session.add(
            Game(
                game_id="mlb_1",
                sport="mlb",
                home_team="New York Yankees",
                away_team="Boston Red Sox",
                kickoff_time=kickoff,
                season=2026,
                home_score=5,
                away_score=3,
            )
        )
        session.add_all(
            [
                OddsSnapshot(
                    game_id="mlb_1",
                    book="pinnacle",
                    market_type="moneyline",
                    odds_home=-150,
                    odds_away=130,
                    captured_at=kickoff - timedelta(hours=1),
                    source="scraped",
                ),
                PaperBet(
                    game_id="mlb_1",
                    book="draftkings",
                    market_type="moneyline",
                    side="home",
                    flagged_at=kickoff - timedelta(hours=2),
                    suggested_stake_pct=1.0,
                    odds_taken=-140,
                    line_taken=None,
                    model_version="market_divergence_v1",
                    edge_pct=3.0,
                    audit_trail={},
                    outcome="pending",
                ),
            ]
        )
        session.commit()

        result = PaperBetSettlementService(session).settle_pending()
        assert result.settled == 1
        bet = session.query(PaperBet).one()
        assert bet.outcome == "win"
        assert bet.clv is not None


class TestManualPlacement:
    def test_places_manual_paper_bet(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=1)
        _seed_upcoming_game(session, game_id="manual_1", kickoff=kickoff)
        session.commit()

        result = ManualBetPlacer(session).place(
            ManualBetRequest(
                game_id="manual_1",
                book="draftkings",
                market_type="spread",
                side="home",
                odds_taken=150,
                line_taken=-3.5,
                stake_pct=1.5,
            )
        )
        session.commit()

        assert result.success is True
        bet = session.query(PaperBet).one()
        assert bet.model_version == "manual_v1"
        assert bet.outcome == "pending"
        assert bet.suggested_stake_pct == pytest.approx(1.5)
        assert bet.audit_trail["placement_type"] == "manual"

    def test_rejects_duplicate_pending_manual_bet(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=1)
        _seed_upcoming_game(session, game_id="manual_1", kickoff=kickoff)
        session.commit()

        placer = ManualBetPlacer(session)
        first = placer.place(
            ManualBetRequest(
                game_id="manual_1",
                book="draftkings",
                market_type="spread",
                side="home",
                odds_taken=150,
                line_taken=-3.5,
            )
        )
        second = placer.place(
            ManualBetRequest(
                game_id="manual_1",
                book="draftkings",
                market_type="spread",
                side="home",
                odds_taken=160,
                line_taken=-3.5,
            )
        )
        session.commit()

        assert first.success is True
        assert second.success is False
        assert second.duplicate is True
        assert session.query(PaperBet).count() == 1

    def test_replace_existing_pending_bet(self, session: Session):
        kickoff = datetime.now(timezone.utc) + timedelta(days=1)
        _seed_upcoming_game(session, game_id="manual_1", kickoff=kickoff)
        session.commit()

        placer = ManualBetPlacer(session)
        first = placer.place(
            ManualBetRequest(
                game_id="manual_1",
                book="draftkings",
                market_type="spread",
                side="home",
                odds_taken=150,
                line_taken=-3.5,
                model_version="old_model",
                edge_pct=40.0,
            )
        )
        second = placer.place(
            ManualBetRequest(
                game_id="manual_1",
                book="draftkings",
                market_type="spread",
                side="home",
                odds_taken=160,
                line_taken=-3.5,
                model_version="new_model",
                edge_pct=2.5,
            ),
            replace_existing=True,
        )
        session.commit()

        assert first.success is True
        assert second.success is True
        assert second.updated is True
        bet = session.query(PaperBet).one()
        assert bet.odds_taken == 160
        assert bet.model_version == "new_model"
        assert bet.edge_pct == pytest.approx(2.5)


class TestMetrics:
    def test_compute_summary_and_clv_trend(self, session: Session):
        now = datetime.now(timezone.utc)
        session.add_all(
            [
                PaperBet(
                    game_id="g1",
                    book="draftkings",
                    market_type="spread",
                    side="home",
                    flagged_at=now - timedelta(days=5),
                    suggested_stake_pct=2.0,
                    odds_taken=-110,
                    line_taken=-3.5,
                    model_version="test",
                    edge_pct=4.0,
                    audit_trail={},
                    outcome="win",
                    clv=0.5,
                    settled_at=now - timedelta(days=4),
                ),
                PaperBet(
                    game_id="g2",
                    book="draftkings",
                    market_type="spread",
                    side="home",
                    flagged_at=now - timedelta(days=3),
                    suggested_stake_pct=2.0,
                    odds_taken=-110,
                    line_taken=-3.5,
                    model_version="test",
                    edge_pct=4.0,
                    audit_trail={},
                    outcome="loss",
                    clv=1.0,
                    settled_at=now - timedelta(days=2),
                ),
                PaperBet(
                    game_id="g3",
                    book="draftkings",
                    market_type="spread",
                    side="home",
                    flagged_at=now - timedelta(days=1),
                    suggested_stake_pct=1.0,
                    odds_taken=-110,
                    line_taken=-3.5,
                    model_version="test",
                    edge_pct=3.0,
                    audit_trail={},
                    outcome="pending",
                ),
            ]
        )
        session.commit()

        bets = session.query(PaperBet).all()
        summary = compute_summary(bets)
        assert summary.total_bets == 3
        assert summary.pending == 1
        assert summary.settled == 2
        assert summary.wins == 1
        assert summary.losses == 1
        assert summary.avg_clv == pytest.approx(0.75)
        assert summary.win_rate == pytest.approx(50.0)

        trend = clv_trend(bets)
        assert len(trend) == 2
        assert trend[-1].cumulative_avg_clv == pytest.approx(0.75)
        assert summary.clv_trend_positive is True


class TestPaperBankroll:
    def test_bet_money_view_pending(self):
        from sports_ev.paper.bankroll import bet_money_view

        bet = PaperBet(
            game_id="g1",
            book="draftkings",
            market_type="moneyline",
            side="home",
            suggested_stake_pct=2.0,
            odds_taken=150,
            line_taken=None,
            model_version="test",
            edge_pct=3.0,
            audit_trail={},
            outcome="pending",
        )
        view = bet_money_view(bet, bankroll_at_bet=1000.0)
        assert view.stake_dollars == pytest.approx(20.0)
        assert view.to_win_dollars == pytest.approx(30.0)
        assert view.payout_dollars == pytest.approx(50.0)
        assert view.profit_loss_dollars is None

    def test_pick_money_preview_zero_stake(self):
        from sports_ev.paper.bankroll import pick_money_preview

        view = pick_money_preview(bankroll=1000.0, stake_pct=0.0, american_odds=-110)
        assert view.stake_dollars == 0.0
        assert view.to_win_dollars == 0.0
        assert view.payout_dollars == 0.0

    def test_compute_bankroll_state_with_settled_win(self):
        from sports_ev.paper.bankroll import compute_bankroll_state

        now = datetime.now(timezone.utc)
        bets = [
            PaperBet(
                id=1,
                game_id="g1",
                book="draftkings",
                market_type="moneyline",
                side="home",
                suggested_stake_pct=2.0,
                odds_taken=-110,
                line_taken=None,
                model_version="test",
                edge_pct=3.0,
                audit_trail={},
                outcome="win",
                flagged_at=now - timedelta(days=2),
            ),
            PaperBet(
                id=2,
                game_id="g2",
                book="draftkings",
                market_type="moneyline",
                side="away",
                suggested_stake_pct=1.0,
                odds_taken=200,
                line_taken=None,
                model_version="test",
                edge_pct=2.5,
                audit_trail={},
                outcome="pending",
                flagged_at=now - timedelta(days=1),
            ),
        ]
        snapshot, views = compute_bankroll_state(bets, starting_bankroll=1000.0)
        assert snapshot.settled_profit > 0
        assert snapshot.at_risk == pytest.approx(views[2].stake_dollars)
        assert views[2].stake_dollars == pytest.approx(snapshot.current_bankroll * 0.01)


class TestSportBreakdown:
    def test_groups_bets_by_sport(self):
        from sports_ev.paper.metrics import compute_sport_breakdown

        now = datetime.now(timezone.utc)
        bets = [
            PaperBet(
                id=1,
                game_id="mlb_1",
                book="draftkings",
                market_type="moneyline",
                side="home",
                suggested_stake_pct=2.0,
                odds_taken=-110,
                line_taken=None,
                model_version="test",
                edge_pct=3.0,
                audit_trail={},
                outcome="win",
                flagged_at=now,
            ),
            PaperBet(
                id=2,
                game_id="nfl_1",
                book="draftkings",
                market_type="spread",
                side="away",
                suggested_stake_pct=1.0,
                odds_taken=-110,
                line_taken=-3.5,
                model_version="test",
                edge_pct=2.5,
                audit_trail={},
                outcome="loss",
                flagged_at=now,
            ),
            PaperBet(
                id=3,
                game_id="mlb_2",
                book="draftkings",
                market_type="moneyline",
                side="home",
                suggested_stake_pct=1.5,
                odds_taken=150,
                line_taken=None,
                model_version="test",
                edge_pct=1.0,
                audit_trail={},
                outcome="pending",
                flagged_at=now,
            ),
        ]
        rows = compute_sport_breakdown(
            bets,
            sport_by_game_id={"mlb_1": "mlb", "mlb_2": "mlb", "nfl_1": "nfl"},
        )
        by_sport = {row.sport: row for row in rows}
        assert set(by_sport) == {"mlb", "nfl"}
        assert by_sport["mlb"].total_bets == 2
        assert by_sport["mlb"].pending == 1
        assert by_sport["mlb"].settled == 1
        assert by_sport["nfl"].losses == 1


class TestEdgeBuckets:
    def test_groups_settled_bets_by_edge_bucket(self):
        from sports_ev.paper.bankroll import compute_bankroll_state
        from sports_ev.paper.metrics import compute_paper_edge_buckets

        now = datetime.now(timezone.utc)
        bets = [
            PaperBet(
                id=1,
                game_id="g1",
                book="draftkings",
                market_type="spread",
                side="home",
                suggested_stake_pct=2.0,
                odds_taken=-110,
                line_taken=-3.5,
                model_version="test",
                edge_pct=1.5,
                audit_trail={},
                outcome="win",
                clv=0.2,
                settled_at=now,
                flagged_at=now,
            ),
            PaperBet(
                id=2,
                game_id="g2",
                book="draftkings",
                market_type="spread",
                side="home",
                suggested_stake_pct=2.0,
                odds_taken=-110,
                line_taken=-3.5,
                model_version="test",
                edge_pct=3.0,
                audit_trail={},
                outcome="loss",
                clv=0.4,
                settled_at=now,
                flagged_at=now,
            ),
            PaperBet(
                id=3,
                game_id="g3",
                book="draftkings",
                market_type="spread",
                side="home",
                suggested_stake_pct=1.0,
                odds_taken=-110,
                line_taken=-3.5,
                model_version="test",
                edge_pct=5.0,
                audit_trail={},
                outcome="pending",
                flagged_at=now,
            ),
        ]
        _, money_views = compute_bankroll_state(bets, starting_bankroll=1000.0)
        rows = compute_paper_edge_buckets(bets, money_views=money_views, min_edge_pct=2.5)
        by_bucket = {row.bucket: row for row in rows}

        assert by_bucket["0-2%"].bets == 1
        assert by_bucket["0-2%"].wins == 1
        assert by_bucket["0-2%"].plus_ev_only is False
        assert by_bucket["2-4%"].bets == 1
        assert by_bucket["2-4%"].losses == 1
        assert by_bucket["2-4%"].plus_ev_only is False
        assert by_bucket["4-6%"].bets == 0
        assert by_bucket["4-6%"].plus_ev_only is True

    def test_edge_bucket_label_boundaries(self):
        from sports_ev.paper.metrics import _edge_bucket_label

        assert _edge_bucket_label(0.0) == "0-2%"
        assert _edge_bucket_label(1.99) == "0-2%"
        assert _edge_bucket_label(2.0) == "2-4%"
        assert _edge_bucket_label(3.99) == "2-4%"
        assert _edge_bucket_label(6.0) == "6%+"

