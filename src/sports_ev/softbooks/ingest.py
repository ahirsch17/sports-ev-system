from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.integrity.circuit_breaker import CircuitBreaker
from sports_ev.integrity.parsing import hash_raw_response
from sports_ev.integrity.validators import OddsSnapshotPayload
from sports_ev.softbooks.base import ScrapeResult, SoftBookScraper
from sports_ev.softbooks.matching import find_game_for_matchup
from sports_ev.softbooks.models import ParsedMoneylineLine, ParsedSpreadLine
from sports_ev.softbooks.registry import default_market_for_sport, get_scraper, list_scrapers


@dataclass
class SoftBookPollResult:
    books_polled: int = 0
    lines_scraped: int = 0
    snapshots_accepted: int = 0
    snapshots_rejected: int = 0
    unmatched_games: int = 0
    errors: list[str] = field(default_factory=list)


class SoftBookIngestService:
    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.circuit_breaker = CircuitBreaker(session, self.settings)

    def parsed_spread_to_payload(
        self,
        line: ParsedSpreadLine,
        *,
        book: str,
        game_id: str,
        parser_version: str,
        raw_hash: str | None = None,
        captured_at: datetime | None = None,
    ) -> OddsSnapshotPayload:
        return OddsSnapshotPayload(
            game_id=game_id,
            book=book,
            market_type="spread",
            captured_at=captured_at or datetime.now(timezone.utc),
            odds_home=line.odds_home,
            odds_away=line.odds_away,
            line=line.line,
            source="scraped",
            parser_version=parser_version,
            raw_response_hash=raw_hash,
        )

    def parsed_moneyline_to_payload(
        self,
        line: ParsedMoneylineLine,
        *,
        book: str,
        game_id: str,
        parser_version: str,
        raw_hash: str | None = None,
        captured_at: datetime | None = None,
    ) -> OddsSnapshotPayload:
        return OddsSnapshotPayload(
            game_id=game_id,
            book=book,
            market_type="moneyline",
            captured_at=captured_at or datetime.now(timezone.utc),
            odds_home=line.odds_home,
            odds_away=line.odds_away,
            line=None,
            source="scraped",
            parser_version=parser_version,
            raw_response_hash=raw_hash,
        )

    def ingest_scrape_result(self, scrape: ScrapeResult) -> SoftBookPollResult:
        result = SoftBookPollResult(books_polled=1)
        raw_hash = hash_raw_response(scrape.raw_response) if scrape.raw_response else None

        for error in scrape.errors:
            result.errors.append(f"{scrape.book}: {error}")

        for line in scrape.parsed_lines:
            result.lines_scraped += 1
            game = find_game_for_matchup(
                self.session,
                sport=scrape.sport,
                home_team=line.home_team,
                away_team=line.away_team,
                kickoff_time=line.kickoff_time,
            )
            if game is None:
                result.unmatched_games += 1
                result.errors.append(
                    f"{scrape.book}: no game match for {line.away_team} @ {line.home_team}"
                )
                continue

            payload = self.parsed_spread_to_payload(
                line,
                book=scrape.book,
                game_id=game.game_id,
                parser_version=scrape.parser_version,
                raw_hash=raw_hash,
                captured_at=scrape.fetched_at,
            )
            ingest = self.circuit_breaker.ingest_odds_snapshot(payload)
            if ingest.accepted:
                result.snapshots_accepted += 1
            else:
                result.snapshots_rejected += 1

        for line in scrape.parsed_moneylines:
            result.lines_scraped += 1
            game = find_game_for_matchup(
                self.session,
                sport=scrape.sport,
                home_team=line.home_team,
                away_team=line.away_team,
                kickoff_time=line.kickoff_time,
            )
            if game is None:
                result.unmatched_games += 1
                result.errors.append(
                    f"{scrape.book}: no game match for {line.away_team} @ {line.home_team}"
                )
                continue

            payload = self.parsed_moneyline_to_payload(
                line,
                book=scrape.book,
                game_id=game.game_id,
                parser_version=scrape.parser_version,
                raw_hash=raw_hash,
                captured_at=scrape.fetched_at,
            )
            ingest = self.circuit_breaker.ingest_odds_snapshot(payload)
            if ingest.accepted:
                result.snapshots_accepted += 1
            else:
                result.snapshots_rejected += 1

        return result

    def poll_books(
        self,
        books: list[str] | None = None,
        *,
        sport: str = "nfl",
        market_type: str | None = None,
    ) -> SoftBookPollResult:
        books = books or list_scrapers()
        market_type = market_type or default_market_for_sport(sport, self.settings)
        aggregate = SoftBookPollResult()

        for book in books:
            scraper = get_scraper(book, self.settings)
            try:
                scrape = scraper.fetch_odds(sport=sport, market_type=market_type)
                book_result = self.ingest_scrape_result(scrape)
                aggregate.books_polled += book_result.books_polled
                aggregate.lines_scraped += book_result.lines_scraped
                aggregate.snapshots_accepted += book_result.snapshots_accepted
                aggregate.snapshots_rejected += book_result.snapshots_rejected
                aggregate.unmatched_games += book_result.unmatched_games
                aggregate.errors.extend(book_result.errors)
            finally:
                scraper.close()

        return aggregate

    def poll_sport(self, sport: str, *, books: list[str] | None = None) -> SoftBookPollResult:
        market_type = default_market_for_sport(sport, self.settings)
        return self.poll_books(books, sport=sport, market_type=market_type)

    def ingest_from_fixture(
        self,
        book: str,
        parsed_lines: list[ParsedSpreadLine],
        *,
        sport: str = "nfl",
        raw_response: dict | None = None,
        parser_version: str | None = None,
    ) -> SoftBookPollResult:
        scrape = ScrapeResult(
            book=book,
            sport=sport,
            market_type="spread",
            parsed_lines=parsed_lines,
            raw_response=raw_response,
            parser_version=parser_version or f"{book}@1.0",
        )
        return self.ingest_scrape_result(scrape)
