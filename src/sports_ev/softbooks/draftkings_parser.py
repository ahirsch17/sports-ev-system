from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sports_ev.integrity.parsing import (
    AdaptiveParser,
    ParseStrategy,
    parse_american_odds,
    parse_spread_line,
)
from sports_ev.softbooks.models import ParsedSpreadLine, ParsedMoneylineLine

PARSER_VERSION = "draftkings@1.1"


def _normalize_unicode_minus(value: str) -> str:
    return value.replace("\u2212", "-")


def _parse_event_teams(event_name: str) -> tuple[str, str]:
    text = event_name.strip()
    for sep in (" @ ", " at ", " vs ", " v "):
        if sep in text:
            away, home = text.split(sep, 1)
            return away.strip(), home.strip()
    raise ValueError(f"Could not parse teams from event name: {event_name!r}")


def _american_odds_parser(raw: Any) -> int:
    if isinstance(raw, str):
        raw = _normalize_unicode_minus(raw)
    return parse_american_odds(raw)


def _decimal_to_american(decimal: float) -> int:
    if decimal >= 2.0:
        return int(round((decimal - 1) * 100))
    return int(round(-100 / (decimal - 1)))


def _line_from_label(label: str) -> float:
    match = re.search(r"([+\-]?\d+(?:\.\d+)?)", _normalize_unicode_minus(label))
    if not match:
        raise ValueError(f"No spread line in label: {label!r}")
    return parse_spread_line(match.group(1))


def _extract_offers_primary(data: dict[str, Any], *, subcategory: str) -> list[dict[str, Any]]:
    categories = data.get("eventGroup", {}).get("offerCategories") or []
    for category in categories:
        if category.get("name") != "Game Lines":
            continue
        for descriptor in category.get("offerSubcategoryDescriptors") or []:
            if descriptor.get("name") != subcategory:
                continue
            offers = (descriptor.get("offerSubcategory") or {}).get("offers") or []
            if offers:
                return offers
    raise KeyError(f"{subcategory} offers not found at primary path")


def _extract_offers_primary_spread(data: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_offers_primary(data, subcategory="Spread")


def _extract_offers_fallback(data: dict[str, Any]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "offers" in node and isinstance(node["offers"], list):
                for offer in node["offers"]:
                    outcomes = offer.get("outcomes") or []
                    if len(outcomes) >= 2:
                        offers.append(offer)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data.get("eventGroup") or data)
    if not offers:
        raise KeyError("No spread-like offers found in fallback walk")
    return offers


def _parse_kickoff(event: dict[str, Any]) -> datetime | None:
    start = event.get("startDate") or event.get("startEventDate")
    if not start:
        return None
    return datetime.fromisoformat(str(start).replace("Z", "+00:00"))


def parse_draftkings_spreads(data: dict[str, Any]) -> list[ParsedSpreadLine]:
    events_by_id = {
        str(event["eventId"]): event
        for event in (data.get("eventGroup", {}).get("events") or [])
        if event.get("eventId") is not None
    }

    offers_parser = AdaptiveParser(
        "offers",
        [
            ParseStrategy("primary_path", "1.0", lambda _: _extract_offers_primary_spread(data)),
            ParseStrategy("fallback_walk", "1.0", lambda _: _extract_offers_fallback(data)),
        ],
    )
    offers = offers_parser.parse(data).value

    lines: list[ParsedSpreadLine] = []
    for offer in offers:
        event = events_by_id.get(str(offer.get("eventId", "")))
        if event is None:
            continue

        away_team, home_team = _parse_event_teams(event.get("name", ""))
        outcomes = offer.get("outcomes") or []
        if len(outcomes) < 2:
            continue

        away_outcome, home_outcome = outcomes[0], outcomes[1]

        line_parser = AdaptiveParser(
            "line",
            [
                ParseStrategy(
                    "outcome_line",
                    "1.0",
                    lambda _: home_outcome.get("line") or home_outcome.get("points"),
                ),
                ParseStrategy(
                    "regex_label",
                    "1.0",
                    lambda _: _line_from_label(str(home_outcome.get("label", ""))),
                ),
            ],
        )
        odds_home = AdaptiveParser(
            "odds_home",
            [
                ParseStrategy(
                    "odds_american",
                    "1.0",
                    lambda _: _american_odds_parser(home_outcome.get("oddsAmerican")),
                ),
                ParseStrategy(
                    "odds_decimal",
                    "1.0",
                    lambda _: _decimal_to_american(float(home_outcome.get("oddsDecimal"))),
                ),
            ],
        ).parse(home_outcome).value
        odds_away = AdaptiveParser(
            "odds_away",
            [
                ParseStrategy(
                    "odds_american",
                    "1.0",
                    lambda _: _american_odds_parser(away_outcome.get("oddsAmerican")),
                ),
                ParseStrategy(
                    "odds_decimal",
                    "1.0",
                    lambda _: _decimal_to_american(float(away_outcome.get("oddsDecimal"))),
                ),
            ],
        ).parse(away_outcome).value

        lines.append(
            ParsedSpreadLine(
                home_team=home_team,
                away_team=away_team,
                odds_home=int(odds_home),
                odds_away=int(odds_away),
                line=float(line_parser.parse(home_outcome).value),
                kickoff_time=_parse_kickoff(event),
                external_event_id=str(event.get("eventId")),
            )
        )

    return lines


def _extract_moneyline_offers_fallback(data: dict[str, Any]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "offers" in node and isinstance(node["offers"], list):
                for offer in node["offers"]:
                    outcomes = offer.get("outcomes") or []
                    if len(outcomes) >= 2 and all(
                        (o.get("line") is None and o.get("points") is None)
                        for o in outcomes
                    ):
                        offers.append(offer)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data.get("eventGroup") or data)
    if not offers:
        raise KeyError("No moneyline-like offers found in fallback walk")
    return offers


def parse_draftkings_moneylines(data: dict[str, Any]) -> list[ParsedMoneylineLine]:
    events_by_id = {
        str(event["eventId"]): event
        for event in (data.get("eventGroup", {}).get("events") or [])
        if event.get("eventId") is not None
    }

    offers_parser = AdaptiveParser(
        "offers",
        [
            ParseStrategy(
                "primary_path",
                "1.0",
                lambda _: _extract_offers_primary(data, subcategory="Moneyline"),
            ),
            ParseStrategy("fallback_walk", "1.0", lambda _: _extract_moneyline_offers_fallback(data)),
        ],
    )
    offers = offers_parser.parse(data).value

    lines: list[ParsedMoneylineLine] = []
    for offer in offers:
        event = events_by_id.get(str(offer.get("eventId", "")))
        if event is None:
            continue

        away_team, home_team = _parse_event_teams(event.get("name", ""))
        outcomes = offer.get("outcomes") or []
        if len(outcomes) < 2:
            continue

        away_outcome, home_outcome = outcomes[0], outcomes[1]
        odds_home = AdaptiveParser(
            "odds_home",
            [
                ParseStrategy(
                    "odds_american",
                    "1.0",
                    lambda _: _american_odds_parser(home_outcome.get("oddsAmerican")),
                ),
                ParseStrategy(
                    "odds_decimal",
                    "1.0",
                    lambda _: _decimal_to_american(float(home_outcome.get("oddsDecimal"))),
                ),
            ],
        ).parse(home_outcome).value
        odds_away = AdaptiveParser(
            "odds_away",
            [
                ParseStrategy(
                    "odds_american",
                    "1.0",
                    lambda _: _american_odds_parser(away_outcome.get("oddsAmerican")),
                ),
                ParseStrategy(
                    "odds_decimal",
                    "1.0",
                    lambda _: _decimal_to_american(float(away_outcome.get("oddsDecimal"))),
                ),
            ],
        ).parse(away_outcome).value

        lines.append(
            ParsedMoneylineLine(
                home_team=home_team,
                away_team=away_team,
                odds_home=int(odds_home),
                odds_away=int(odds_away),
                kickoff_time=_parse_kickoff(event),
                external_event_id=str(event.get("eventId")),
            )
        )

    return lines
