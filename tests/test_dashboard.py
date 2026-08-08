from __future__ import annotations



from unittest.mock import MagicMock, patch



from sports_ev.dashboard.actions import ActionResult, get_system_status

from sports_ev.dashboard.theme import APP_TITLE, APP_TAGLINE, hero_html





class TestDashboardActions:

    def test_get_system_status_without_db(self):

        settings = MagicMock()

        settings.database_url = "postgresql+psycopg://bad:bad@localhost:59999/nope"



        with patch("sports_ev.dashboard.actions.session_scope") as mock_scope:

            mock_scope.side_effect = Exception("connection refused")

            status = get_system_status(settings=settings)



        assert status.database_ok is False

        assert "connection refused" in status.message



    def test_action_result_structure(self):

        result = ActionResult(True, "Test", "ok", {"count": 1})

        assert result.success is True

        assert result.details["count"] == 1





class TestDashboardHelpCopy:
    def test_log_plus_ev_only_mentions_threshold(self):
        from sports_ev.config import Settings
        from sports_ev.dashboard.help_copy import log_plus_ev_only_help

        help_text = log_plus_ev_only_help(Settings(min_edge_pct=2.5))
        assert "2.5%" in help_text
        assert "Skips" in help_text

    def test_sidebar_log_matches_log_all(self):
        from sports_ev.config import Settings
        from sports_ev.dashboard.help_copy import log_all_picks_help, sidebar_log_model_picks_help

        settings = Settings(min_edge_pct=2.0)
        assert "+EV" in sidebar_log_model_picks_help(settings)
        assert "Log all picks" in sidebar_log_model_picks_help(settings)
        assert "2.0%" in log_all_picks_help(settings)

    def test_get_default_sport_prefers_pending_bets(self):
        from datetime import datetime, timezone

        from sports_ev.config import Settings
        from sports_ev.dashboard.actions import get_default_sport
        from sports_ev.db.models import Base, Game, PaperBet
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        now = datetime.now(timezone.utc)
        db.add(
            Game(
                game_id="mlb_1",
                sport="mlb",
                home_team="Padres",
                away_team="Dodgers",
                kickoff_time=now,
                season=2026,
            )
        )
        db.add(
            Game(
                game_id="nfl_1",
                sport="nfl",
                home_team="Chiefs",
                away_team="Bills",
                kickoff_time=now,
                season=2026,
            )
        )
        db.add(
            PaperBet(
                game_id="mlb_1",
                book="draftkings",
                market_type="moneyline",
                side="home",
                flagged_at=now,
                suggested_stake_pct=1.0,
                odds_taken=-110,
                model_version="test",
                edge_pct=2.0,
                audit_trail={},
                outcome="pending",
            )
        )
        db.commit()
        assert get_default_sport(settings=Settings(), pending_bets=db.query(PaperBet).all(), session=db) == "mlb"


    def test_paper_bankroll_help_mentions_fake_money(self):
        from sports_ev.dashboard.help_copy import PAPER_BANKROLL_HELP

        assert "fake money" in PAPER_BANKROLL_HELP.lower()
        assert "Not real money" in PAPER_BANKROLL_HELP


class TestDashboardTheme:

    def test_branding_constants(self):

        assert APP_TITLE == "SportsPredictor"

        assert "Research only" in APP_TAGLINE



    def test_hero_html_includes_brand_and_sport(self):

        html = hero_html(sport_label="MLB")

        assert "SportsPredictor" in html

        assert "MLB" in html

        assert "sp-brand-mark" in html

        assert "Research only" in html

    def test_compact_header(self):
        from sports_ev.dashboard.theme import compact_header_html

        html = compact_header_html(sport_label="MLB")
        assert "SportsPredictor" in html
        assert "MLB" in html
        assert "sp-compact-header" in html


