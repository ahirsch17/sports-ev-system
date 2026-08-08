"""Self-built soft-book scrapers (DraftKings, FanDuel) with adaptive parsing."""

from sports_ev.softbooks.ingest import SoftBookIngestService, SoftBookPollResult
from sports_ev.softbooks.registry import get_scraper, list_scrapers

__all__ = ["SoftBookIngestService", "SoftBookPollResult", "get_scraper", "list_scrapers"]
