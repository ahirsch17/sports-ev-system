from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sports_ev.integrity.parsing import AdaptiveParser, ParseStrategy, parse_american_odds
from sports_ev.softbooks.models import ParsedSpreadLine, ParsedMoneylineLine

PARSER_VERSION = "fanduel@1.1"


def _extract_handicap_markets_primary(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}
    markets = attachments.get("markets") or {}
    results: list[tuple[str, dict[str, Any]]] = []

    for market_id, market in markets.items():
        market_type = str(market.get("marketType", ""))
        if "HANDICAP" not in market_type.upper():
            continue
        event_id = str(market.get("eventId", ""))
        if event_id and event_id in events:
            results.append((event_id, market))
    if not results:
        raise KeyError("No handicap markets found at primary path")
    return results


def _extract_handicap_markets_fallback(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}
    markets = attachments.get("markets") or {}
    results: list[tuple[str, dict[str, Any]]] = []

    for market_id, market in markets.items():
        runners = market.get("runners") or []
        if len(runners) < 2:
            continue
        has_handicap = any(r.get("handicap") is not None for r in runners)
        if not has_handicap:
            continue
        event_id = str(market.get("eventId", ""))
        if event_id in events:
            results.append((event_id, market))
    if not results:
        raise KeyError("No handicap markets found in fallback")
    return results


def _parse_kickoff(event: dict[str, Any]) -> datetime | None:
    start = event.get("openDate") or event.get("startTime")
    if not start:
        return None
    if isinstance(start, (int, float)):
        return datetime.fromtimestamp(start / 1000.0, tz=timezone.utc)
    return datetime.fromisoformat(str(start).replace("Z", "+00:00"))


def _runner_odds(runner: dict[str, Any]) -> int:
    odds = ((runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get(
        "americanOdds"
    )
    if odds is None:
        odds = runner.get("americanOdds")
    if odds is None:
        decimal = (runner.get("winRunnerOdds") or {}).get("trueOdds", {}).get("decimalOdds")
        if decimal:
            dec = float(decimal.get("decimalOdds", decimal) if isinstance(decimal, dict) else decimal)
            if dec >= 2.0:
                return int(round((dec - 1) * 100))
            return int(round(-100 / (dec - 1)))
    return parse_american_odds(str(odds))


def parse_fanduel_spreads(data: dict[str, Any]) -> list[ParsedSpreadLine]:
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}

    markets_parser = AdaptiveParser(
        "markets",
        [
            ParseStrategy("primary_path", "1.0", lambda _: _extract_handicap_markets_primary(data)),
            ParseStrategy("fallback_walk", "1.0", lambda _: _extract_handicap_markets_fallback(data)),
        ],
    )
    market_pairs = markets_parser.parse(data).value

    lines: list[ParsedSpreadLine] = []
    for event_id, market in market_pairs:
        event = events.get(event_id) or {}
        name = event.get("name", "")
        if " @ " in name:
            away_team, home_team = [part.strip() for part in name.split(" @ ", 1)]
        elif " v " in name:
            away_team, home_team = [part.strip() for part in name.split(" v ", 1)]
        else:
            continue

        runners = market.get("runners") or []
        if len(runners) < 2:
            continue

        home_runner = runners[0]
        away_runner = runners[1]
        for runner in runners:
            role = str(runner.get("runnerName", "")).lower()
            if home_team.lower() in role:
                home_runner = runner
            elif away_team.lower() in role:
                away_runner = runner

        line = float(home_runner.get("handicap") or away_runner.get("handicap") or 0.0)

        lines.append(
            ParsedSpreadLine(
                home_team=home_team,
                away_team=away_team,
                odds_home=_runner_odds(home_runner),
                odds_away=_runner_odds(away_runner),
                line=line,
                kickoff_time=_parse_kickoff(event),
                external_event_id=event_id,
            )
        )

    return lines


def _extract_moneyline_markets_primary(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}
    markets = attachments.get("markets") or {}
    results: list[tuple[str, dict[str, Any]]] = []

    for market_id, market in markets.items():
        market_type = str(market.get("marketType", "")).upper()
        if "MATCH" not in market_type and "WINNER" not in market_type and "MONEYLINE" not in market_type:
            continue
        runners = market.get("runners") or []
        if any(r.get("handicap") is not None for r in runners):
            continue
        event_id = str(market.get("eventId", ""))
        if event_id and event_id in events:
            results.append((event_id, market))
    if not results:
        raise KeyError("No moneyline markets found at primary path")
    return results


def _extract_moneyline_markets_fallback(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}
    markets = attachments.get("markets") or {}
    results: list[tuple[str, dict[str, Any]]] = []

    for market_id, market in markets.items():
        runners = market.get("runners") or []
        if len(runners) < 2:
            continue
        if any(r.get("handicap") is not None for r in runners):
            continue
        event_id = str(market.get("eventId", ""))
        if event_id in events:
            results.append((event_id, market))
    if not results:
        raise KeyError("No moneyline markets found in fallback")
    return results


def _teams_from_event_name(name: str) -> tuple[str, str] | None:
    if " @ " in name:
        away_team, home_team = [part.strip() for part in name.split(" @ ", 1)]
        return away_team, home_team
    if " v " in name:
        away_team, home_team = [part.strip() for part in name.split(" v ", 1)]
        return away_team, home_team
    return None


def parse_fanduel_moneylines(data: dict[str, Any]) -> list[ParsedMoneylineLine]:
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}

    markets_parser = AdaptiveParser(
        "markets",
        [
            ParseStrategy("primary_path", "1.0", lambda _: _extract_moneyline_markets_primary(data)),
            ParseStrategy("fallback_walk", "1.0", lambda _: _extract_moneyline_markets_fallback(data)),
        ],
    )
    market_pairs = markets_parser.parse(data).value

    lines: list[ParsedMoneylineLine] = []
    for event_id, market in market_pairs:
        event = events.get(event_id) or {}
        teams = _teams_from_event_name(event.get("name", ""))
        if teams is None:
            continue
        away_team, home_team = teams

        runners = market.get("runners") or []
        if len(runners) < 2:
            continue

        home_runner = runners[0]
        away_runner = runners[1]
        for runner in runners:
            role = str(runner.get("runnerName", "")).lower()
            if home_team.lower() in role:
                home_runner = runner
            elif away_team.lower() in role:
                away_runner = runner

        lines.append(
            ParsedMoneylineLine(
                home_team=home_team,
                away_team=away_team,
                odds_home=_runner_odds(home_runner),
                odds_away=_runner_odds(away_runner),
                kickoff_time=_parse_kickoff(event),
                external_event_id=event_id,
            )
        )

    return lines
