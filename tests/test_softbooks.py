from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, OddsSnapshot
from sports_ev.softbooks.draftkings_parser import parse_draftkings_spreads
from sports_ev.softbooks.fanduel_parser import parse_fanduel_spreads
from sports_ev.softbooks.ingest import SoftBookIngestService
from sports_ev.softbooks.matching import find_game_for_matchup, normalize_team, teams_match

FIXTURES = Path(__file__).parent / "fixtures"


class TestTeamMatching:
    def test_teams_match_substring(self):
        assert teams_match("Kansas City Chiefs", "KC Chiefs") is True
        assert teams_match("Baltimore Ravens", "Ravens") is True

    def test_normalize_team(self):
        assert normalize_team("The Kansas City Chiefs") == "kansas city chiefs"


class TestDraftKingsParser:
    def test_parse_spread_primary_path(self):
        raw = json.loads((FIXTURES / "draftkings_spread.json").read_text())
        lines = parse_draftkings_spreads(raw)
        assert len(lines) == 1
        line = lines[0]
        assert line.home_team == "Kansas City Chiefs"
        assert line.away_team == "Baltimore Ravens"
        assert line.line == pytest.approx(-3.5)
        assert line.odds_home == -115
        assert line.odds_away == 100

    def test_unicode_minus_sign(self):
        raw = json.loads((FIXTURES / "draftkings_spread.json").read_text())
        lines = parse_draftkings_spreads(raw)
        assert lines[0].odds_home == -115


class TestFanDuelParser:
    def test_parse_handicap_market(self):
        raw = json.loads((FIXTURES / "fanduel_spread.json").read_text())
        lines = parse_fanduel_spreads(raw)
        assert len(lines) == 1
        line = lines[0]
        assert line.home_team == "Kansas City Chiefs"
        assert line.away_team == "Baltimore Ravens"
        assert line.line == pytest.approx(-3.5)
        assert line.odds_home == -110
        assert line.odds_away == -110


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sess = factory()
    sess.add(
        Game(
            game_id="nfl_test_game",
            sport="nfl",
            home_team="Kansas City Chiefs",
            away_team="Baltimore Ravens",
            kickoff_time=datetime(2026, 9, 10, 0, 20, tzinfo=timezone.utc),
            season=2026,
            week=1,
        )
    )
    sess.commit()
    yield sess
    sess.close()


from sports_ev.sports.registry import get_sport_config, list_sports


class TestSportRegistry:
    def test_list_sports(self):
        assert list_sports() == ["nfl", "mlb"]

    def test_mlb_defaults_to_moneyline(self):
        config = get_sport_config("mlb")
        assert config.default_market == "moneyline"
        assert config.game_id_from_matchup(123) == "mlb_pin_123"
    def test_find_game_for_matchup(self, session: Session):
        game = find_game_for_matchup(
            session,
            home_team="Kansas City Chiefs",
            away_team="Baltimore Ravens",
        )
        assert game is not None
        assert game.game_id == "nfl_test_game"

    def test_ingest_draftkings_through_circuit_breaker(self, session: Session):
        raw = json.loads((FIXTURES / "draftkings_spread.json").read_text())
        lines = parse_draftkings_spreads(raw)
        service = SoftBookIngestService(
            session, Settings(circuit_breaker_max_stale_seconds=86400)
        )
        result = service.ingest_from_fixture("draftkings", lines, raw_response=raw)
        session.commit()

        assert result.snapshots_accepted == 1
        snapshot = session.query(OddsSnapshot).one()
        assert snapshot.book == "draftkings"
        assert snapshot.source == "scraped"
        assert snapshot.line == pytest.approx(-3.5)

    def test_ingest_fanduel_through_circuit_breaker(self, session: Session):
        raw = json.loads((FIXTURES / "fanduel_spread.json").read_text())
        lines = parse_fanduel_spreads(raw)
        service = SoftBookIngestService(
            session, Settings(circuit_breaker_max_stale_seconds=86400)
        )
        result = service.ingest_from_fixture("fanduel", lines, raw_response=raw)
        session.commit()

        assert result.snapshots_accepted == 1

    def test_malformed_odds_rejected_by_circuit_breaker(self, session: Session):
        from sports_ev.softbooks.models import ParsedSpreadLine

        bad_line = ParsedSpreadLine(
            home_team="Kansas City Chiefs",
            away_team="Baltimore Ravens",
            odds_home=999999,
            odds_away=-110,
            line=-3.5,
        )
        service = SoftBookIngestService(session, Settings())
        result = service.ingest_from_fixture("draftkings", [bad_line])
        session.commit()

        assert result.snapshots_accepted == 0
        assert result.snapshots_rejected == 1
        assert session.query(OddsSnapshot).count() == 0

    @patch("sports_ev.softbooks.registry.FanDuelScraper.fetch_odds")
    @patch("sports_ev.softbooks.registry.DraftKingsScraper.fetch_odds")
    def test_poll_books(self, mock_dk, mock_fd, session: Session):
        from sports_ev.softbooks.base import ScrapeResult

        dk_raw = json.loads((FIXTURES / "draftkings_spread.json").read_text())
        fd_raw = json.loads((FIXTURES / "fanduel_spread.json").read_text())
        mock_dk.return_value = ScrapeResult(
            book="draftkings",
            sport="nfl",
            market_type="spread",
            parsed_lines=parse_draftkings_spreads(dk_raw),
            parser_version="draftkings@1.0",
            raw_response=dk_raw,
        )
        mock_fd.return_value = ScrapeResult(
            book="fanduel",
            sport="nfl",
            market_type="spread",
            parsed_lines=parse_fanduel_spreads(fd_raw),
            parser_version="fanduel@1.0",
            raw_response=fd_raw,
        )

        service = SoftBookIngestService(
            session, Settings(circuit_breaker_max_stale_seconds=86400)
        )
        result = service.poll_books(["draftkings", "fanduel"])
        session.commit()

        assert result.books_polled == 2
        assert result.snapshots_accepted == 2
        assert session.query(OddsSnapshot).count() == 2
