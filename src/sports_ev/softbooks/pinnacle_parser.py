from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sports_ev.integrity.parsing import AdaptiveParser, ParseStrategy, parse_american_odds, parse_spread_line
from sports_ev.softbooks.models import ParsedSpreadLine, ParsedMoneylineLine

PARSER_VERSION = "pinnacle@1.1"
PINNACLE_NFL_LEAGUE_ID = 889
PINNACLE_FOOTBALL_SPORT_ID = 15


def _normalize_american(raw: Any) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    return parse_american_odds(str(raw))


def _participants_by_alignment(matchup: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for participant in matchup.get("participants") or []:
        alignment = participant.get("alignment")
        name = participant.get("name")
        if alignment and name:
            result[str(alignment)] = str(name)
    return result


def _parse_kickoff(matchup: dict[str, Any]) -> datetime | None:
    start = matchup.get("startTime")
    if not start:
        return None
    dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _main_spread_markets(markets: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Pick primary (non-alternate) full-game spread per matchup."""
    chosen: dict[int, dict[str, Any]] = {}
    for market in markets:
        if market.get("type") != "spread":
            continue
        if market.get("period", 0) != 0:
            continue
        if market.get("isAlternate"):
            continue
        matchup_id = market.get("matchupId")
        if matchup_id is None:
            continue
        chosen[int(matchup_id)] = market
    return chosen


def parse_pinnacle_spreads(
    matchups: list[dict[str, Any]],
    markets: list[dict[str, Any]],
) -> list[ParsedSpreadLine]:
    matchup_by_id = {
        int(m["id"]): m
        for m in matchups
        if m.get("id") is not None and m.get("type") == "matchup" and not m.get("isLive")
    }
    spread_by_matchup = _main_spread_markets(markets)

    lines: list[ParsedSpreadLine] = []
    for matchup_id, market in spread_by_matchup.items():
        matchup = matchup_by_id.get(matchup_id)
        if matchup is None:
            continue

        teams = _participants_by_alignment(matchup)
        home_team = teams.get("home")
        away_team = teams.get("away")
        if not home_team or not away_team:
            continue

        price_by_designation = {
            str(p.get("designation")): p for p in (market.get("prices") or []) if p.get("designation")
        }
        home_price = price_by_designation.get("home")
        away_price = price_by_designation.get("away")
        if not home_price or not away_price:
            continue

        line_parser = AdaptiveParser(
            "line",
            [
                ParseStrategy("home_points", "1.0", lambda _: home_price.get("points")),
                ParseStrategy(
                    "regex_key",
                    "1.0",
                    lambda _: _line_from_key(str(market.get("key", ""))),
                ),
            ],
        )
        odds_home = AdaptiveParser(
            "odds_home",
            [ParseStrategy("american", "1.0", lambda _: _normalize_american(home_price.get("price")))],
        ).parse(home_price).value
        odds_away = AdaptiveParser(
            "odds_away",
            [ParseStrategy("american", "1.0", lambda _: _normalize_american(away_price.get("price")))],
        ).parse(away_price).value

        lines.append(
            ParsedSpreadLine(
                home_team=home_team,
                away_team=away_team,
                odds_home=int(odds_home),
                odds_away=int(odds_away),
                line=float(line_parser.parse(home_price).value),
                kickoff_time=_parse_kickoff(matchup),
                external_event_id=str(matchup_id),
            )
        )

    return lines


def _main_moneyline_markets(markets: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    chosen: dict[int, dict[str, Any]] = {}
    for market in markets:
        if market.get("type") != "moneyline":
            continue
        if market.get("period", 0) != 0:
            continue
        if market.get("isAlternate"):
            continue
        matchup_id = market.get("matchupId")
        if matchup_id is None:
            continue
        chosen[int(matchup_id)] = market
    return chosen


def parse_pinnacle_moneylines(
    matchups: list[dict[str, Any]],
    markets: list[dict[str, Any]],
) -> list[ParsedMoneylineLine]:
    matchup_by_id = {
        int(m["id"]): m
        for m in matchups
        if m.get("id") is not None and m.get("type") == "matchup" and not m.get("isLive")
    }
    moneyline_by_matchup = _main_moneyline_markets(markets)

    lines: list[ParsedMoneylineLine] = []
    for matchup_id, market in moneyline_by_matchup.items():
        matchup = matchup_by_id.get(matchup_id)
        if matchup is None:
            continue

        teams = _participants_by_alignment(matchup)
        home_team = teams.get("home")
        away_team = teams.get("away")
        if not home_team or not away_team:
            continue

        price_by_designation = {
            str(p.get("designation")): p for p in (market.get("prices") or []) if p.get("designation")
        }
        home_price = price_by_designation.get("home")
        away_price = price_by_designation.get("away")
        if not home_price or not away_price:
            continue

        odds_home = AdaptiveParser(
            "odds_home",
            [ParseStrategy("american", "1.0", lambda _: _normalize_american(home_price.get("price")))],
        ).parse(home_price).value
        odds_away = AdaptiveParser(
            "odds_away",
            [ParseStrategy("american", "1.0", lambda _: _normalize_american(away_price.get("price")))],
        ).parse(away_price).value

        lines.append(
            ParsedMoneylineLine(
                home_team=home_team,
                away_team=away_team,
                odds_home=int(odds_home),
                odds_away=int(odds_away),
                kickoff_time=_parse_kickoff(matchup),
                external_event_id=str(matchup_id),
            )
        )

    return lines


def _line_from_key(key: str) -> float:
    match = re.search(r";s;([+\-]?\d+(?:\.\d+)?)", key)
    if not match:
        raise ValueError(f"No spread in market key: {key!r}")
    return parse_spread_line(match.group(1))
