from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from sports_ev.config import Settings, get_settings
from sports_ev.integrity.validators import OddsSnapshotPayload


from sports_ev.softbooks.models import ParsedSpreadLine, ParsedMoneylineLine


class SoftBookScraperError(Exception):
    pass


@dataclass
class ScrapeResult:
    book: str
    market_type: str = "spread"
    sport: str = "nfl"
    parsed_lines: list[ParsedSpreadLine] = field(default_factory=list)
    parsed_moneylines: list[ParsedMoneylineLine] = field(default_factory=list)
    parser_version: str = "1.0"
    raw_response: Any | None = None
    errors: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RateLimitedClient:
    """Simple HTTP client with minimum interval between requests."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = 2.0,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        self.min_interval_seconds = min_interval_seconds
        self._last_request = 0.0
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; SportsEVResearch/0.1; +research/paper-tracking)"
                ),
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        response = self._client.get(url, params=params)
        self._last_request = time.monotonic()

        if response.status_code >= 400:
            raise SoftBookScraperError(
                f"HTTP {response.status_code} from {url}: {response.text[:200]}"
            )
        return response.json()


class SoftBookScraper(ABC):
    book: str
    parser_version: str

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http: RateLimitedClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.http = http or RateLimitedClient(
            min_interval_seconds=self.settings.softbook_min_interval_seconds
        )

    @abstractmethod
    def fetch_odds(self, *, sport: str = "nfl", market_type: str = "spread") -> ScrapeResult:
        """Fetch current odds snapshots from the book."""

    def fetch_spreads(self, *, sport: str = "nfl") -> ScrapeResult:
        return self.fetch_odds(sport=sport, market_type="spread")

    def close(self) -> None:
        self.http.close()
