from __future__ import annotations

from typing import Callable

from sports_ev.config import Settings, get_settings
from sports_ev.softbooks.base import RateLimitedClient, ScrapeResult, SoftBookScraper, SoftBookScraperError
from sports_ev.softbooks.draftkings_parser import PARSER_VERSION as DK_PARSER_VERSION
from sports_ev.softbooks.draftkings_parser import parse_draftkings_moneylines, parse_draftkings_spreads
from sports_ev.softbooks.fanduel_parser import PARSER_VERSION as FD_PARSER_VERSION
from sports_ev.softbooks.fanduel_parser import parse_fanduel_moneylines, parse_fanduel_spreads
from sports_ev.softbooks.espn_odds import PARSER_VERSION as ESPN_DK_PARSER_VERSION
from sports_ev.softbooks.espn_odds import fetch_espn_draftkings_odds
from sports_ev.softbooks.pinnacle_parser import PARSER_VERSION as PINNACLE_PARSER_VERSION
from sports_ev.softbooks.pinnacle_parser import parse_pinnacle_moneylines, parse_pinnacle_spreads
from sports_ev.sports.registry import SportConfig, get_sport_config


class DraftKingsScraper(SoftBookScraper):
    book = "draftkings"
    parser_version = DK_PARSER_VERSION

    def fetch_odds(self, *, sport: str = "nfl", market_type: str = "spread") -> ScrapeResult:
        result = ScrapeResult(book=self.book, sport=sport, market_type=market_type)
        config = get_sport_config(sport, self.settings)
        try:
            group_id = config.draftkings_event_group_id
            base = (
                f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{group_id}"
            )
            overview = self.http.get_json(f"{base}/?format=json")
            categories = overview.get("eventGroup", {}).get("offerCategories") or []
            game_lines = next((c for c in categories if c.get("name") == "Game Lines"), None)
            if game_lines is None:
                raise SoftBookScraperError("DraftKings Game Lines category not found")

            category_id = game_lines["offerCategoryId"]
            category_data = self.http.get_json(f"{base}/categories/{category_id}?format=json")
            descriptors = (
                category_data.get("eventGroup", {})
                .get("offerCategories", [{}])[0]
                .get("offerSubcategoryDescriptors")
                or []
            )
            subcategory_name = "Moneyline" if market_type == "moneyline" else "Spread"
            sub_desc = next((d for d in descriptors if d.get("name") == subcategory_name), None)
            if sub_desc is None:
                raise SoftBookScraperError(f"DraftKings {subcategory_name} subcategory not found")

            subcategory_id = sub_desc["subcategoryId"]
            data = self.http.get_json(
                f"{base}/categories/{category_id}/subcategories/{subcategory_id}?format=json"
            )
            result.raw_response = data
            result.parser_version = self.parser_version
            if market_type == "moneyline":
                result.parsed_moneylines = parse_draftkings_moneylines(data)
            else:
                result.parsed_lines = parse_draftkings_spreads(data)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))

        if not result.parsed_moneylines and not result.parsed_lines:
            ml, spreads, raw_payloads, espn_errors = fetch_espn_draftkings_odds(
                sport=sport,
                market_type=market_type,
                days_ahead=7,
                http=None,
            )
            result.errors.extend(espn_errors)
            if raw_payloads:
                result.raw_response = {"espn_scoreboards": raw_payloads}
                result.parser_version = ESPN_DK_PARSER_VERSION
            if market_type == "moneyline":
                result.parsed_moneylines = ml
            else:
                result.parsed_lines = spreads
            if ml or spreads:
                result.errors = [
                    "DraftKings direct API blocked. Using ESPN syndicated DraftKings lines instead."
                ]
        return result


class FanDuelScraper(SoftBookScraper):
    book = "fanduel"
    parser_version = FD_PARSER_VERSION

    def fetch_odds(self, *, sport: str = "nfl", market_type: str = "spread") -> ScrapeResult:
        result = ScrapeResult(book=self.book, sport=sport, market_type=market_type)
        config = get_sport_config(sport, self.settings)
        try:
            url = f"{self.settings.fanduel_api_base_url.rstrip('/')}/api/content-managed-page"
            params = {
                "page": "CUSTOM",
                "customPageId": config.fanduel_page_id,
                "timezone": "America/New_York",
            }
            data = self.http.get_json(url, params=params)
            result.raw_response = data
            result.parser_version = self.parser_version
            if market_type == "moneyline":
                result.parsed_moneylines = parse_fanduel_moneylines(data)
            else:
                result.parsed_lines = parse_fanduel_spreads(data)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))
        return result


class PinnacleScraper(SoftBookScraper):
    """Scrape Pinnacle lines via the public guest JSON API (no API key)."""

    book = "pinnacle"
    parser_version = PINNACLE_PARSER_VERSION
    BASE_URL = "https://guest.api.arcadia.pinnacle.com/0.1"

    def fetch_odds(self, *, sport: str = "nfl", market_type: str = "spread") -> ScrapeResult:
        result = ScrapeResult(book=self.book, sport=sport, market_type=market_type)
        config = get_sport_config(sport, self.settings)
        try:
            league_id = config.pinnacle_league_id
            matchups = self.http.get_json(f"{self.BASE_URL}/leagues/{league_id}/matchups")
            markets = self.http.get_json(f"{self.BASE_URL}/leagues/{league_id}/markets/straight")
            result.raw_response = {"matchups": matchups, "markets": markets}
            result.parser_version = self.parser_version
            if market_type == "moneyline":
                result.parsed_moneylines = parse_pinnacle_moneylines(matchups, markets)
            else:
                result.parsed_lines = parse_pinnacle_spreads(matchups, markets)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))
        return result


def get_scraper(book: str, settings: Settings | None = None) -> SoftBookScraper:
    settings = settings or get_settings()
    scrapers: dict[str, Callable[[], SoftBookScraper]] = {
        "draftkings": lambda: DraftKingsScraper(settings),
        "fanduel": lambda: FanDuelScraper(settings),
        "pinnacle": lambda: PinnacleScraper(settings),
    }
    key = book.lower().strip()
    if key not in scrapers:
        raise SoftBookScraperError(f"Unknown soft book: {book}")
    return scrapers[key]()


def list_scrapers(*, include_sharp: bool = False) -> list[str]:
    books = ["draftkings", "fanduel"]
    if include_sharp:
        books.append("pinnacle")
    return books


def default_market_for_sport(sport: str, settings: Settings | None = None) -> str:
    return get_sport_config(sport, settings or get_settings()).default_market
