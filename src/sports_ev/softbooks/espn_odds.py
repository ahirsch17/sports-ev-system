from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from sports_ev.scores.espn import ESPN_BASE, EspnScoreClient
from sports_ev.softbooks.models import ParsedMoneylineLine, ParsedSpreadLine
from sports_ev.sports.registry import get_sport_config

PARSER_VERSION = "espn_draftkings@1.0"


def parse_american_odds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not text or text in {"-", "0", "+0", "-0"}:
        return None
    upper = text.upper()
    if upper in {"EVEN", "EV", "PK", "PICK"}:
        return 100
    try:
        return int(text.replace("+", ""))
    except ValueError:
        return None


def parse_spread_line(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not text or text in {"-", "PK", "PICK"}:
        return None
    try:
        return float(text.replace("+", ""))
    except ValueError:
        return None


def _competitor_teams(competition: dict[str, Any]) -> tuple[str | None, str | None]:
    home_team: str | None = None
    away_team: str | None = None
    for competitor in competition.get("competitors") or []:
        team = (competitor.get("team") or {}).get("displayName") or (
            competitor.get("team") or {}
        ).get("name")
        if not team:
            continue
        side = str(competitor.get("homeAway", "")).lower()
        if side == "home":
            home_team = str(team)
        elif side == "away":
            away_team = str(team)
    return home_team, away_team


def _event_kickoff(event: dict[str, Any]) -> datetime | None:
    raw = event.get("date")
    if not raw:
        return None
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _draftkings_odds_block(competition: dict[str, Any]) -> dict[str, Any] | None:
    for block in competition.get("odds") or []:
        provider = (block.get("provider") or {}).get("name") or ""
        if provider.lower() == "draftkings":
            return block
    return None


def parse_espn_draftkings_moneylines(payload: dict[str, Any]) -> list[ParsedMoneylineLine]:
    lines: list[ParsedMoneylineLine] = []
    for event in payload.get("events") or []:
        kickoff = _event_kickoff(event)
        for competition in event.get("competitions") or []:
            status = ((competition.get("status") or {}).get("type") or {}).get("state")
            if status not in {"pre", "in"}:
                continue
            home_team, away_team = _competitor_teams(competition)
            if not home_team or not away_team:
                continue
            block = _draftkings_odds_block(competition)
            if block is None:
                continue
            moneyline = block.get("moneyline") or {}
            home_odds = parse_american_odds((moneyline.get("home") or {}).get("close", {}).get("odds"))
            away_odds = parse_american_odds((moneyline.get("away") or {}).get("close", {}).get("odds"))
            if home_odds is None or away_odds is None:
                continue
            lines.append(
                ParsedMoneylineLine(
                    home_team=home_team,
                    away_team=away_team,
                    odds_home=home_odds,
                    odds_away=away_odds,
                    kickoff_time=kickoff,
                    external_event_id=str(event.get("id")) if event.get("id") is not None else None,
                )
            )
    return lines


def parse_espn_draftkings_spreads(payload: dict[str, Any]) -> list[ParsedSpreadLine]:
    lines: list[ParsedSpreadLine] = []
    for event in payload.get("events") or []:
        kickoff = _event_kickoff(event)
        for competition in event.get("competitions") or []:
            status = ((competition.get("status") or {}).get("type") or {}).get("state")
            if status not in {"pre", "in"}:
                continue
            home_team, away_team = _competitor_teams(competition)
            if not home_team or not away_team:
                continue
            block = _draftkings_odds_block(competition)
            if block is None:
                continue
            point_spread = block.get("pointSpread") or {}
            home_close = (point_spread.get("home") or {}).get("close") or {}
            away_close = (point_spread.get("away") or {}).get("close") or {}
            line = parse_spread_line(home_close.get("line"))
            home_odds = parse_american_odds(home_close.get("odds"))
            away_odds = parse_american_odds(away_close.get("odds"))
            if line is None or home_odds is None or away_odds is None:
                continue
            lines.append(
                ParsedSpreadLine(
                    home_team=home_team,
                    away_team=away_team,
                    odds_home=home_odds,
                    odds_away=away_odds,
                    line=line,
                    kickoff_time=kickoff,
                    external_event_id=str(event.get("id")) if event.get("id") is not None else None,
                )
            )
    return lines


def fetch_espn_draftkings_odds(
    *,
    sport: str,
    market_type: str,
    days_ahead: int = 7,
    http: EspnScoreClient | None = None,
) -> tuple[list[ParsedMoneylineLine], list[ParsedSpreadLine], list[dict[str, Any]], list[str]]:
    """Fetch DraftKings lines syndicated through ESPN scoreboard odds."""
    config = get_sport_config(sport)
    client = http or EspnScoreClient()
    own_client = http is None
    errors: list[str] = []
    raw_payloads: list[dict[str, Any]] = []
    moneylines: list[ParsedMoneylineLine] = []
    spreads: list[ParsedSpreadLine] = []

    try:
        today = datetime.now(timezone.utc).date()
        for offset in range(days_ahead + 1):
            on_date = today + timedelta(days=offset)
            try:
                payload = client.fetch_scoreboard(config.espn_scoreboard_path, on_date=on_date)
                raw_payloads.append(payload)
                if market_type == "moneyline":
                    moneylines.extend(parse_espn_draftkings_moneylines(payload))
                else:
                    spreads.extend(parse_espn_draftkings_spreads(payload))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ESPN {on_date.isoformat()}: {exc}")
    finally:
        if own_client:
            client.close()

    return moneylines, spreads, raw_payloads, errors
