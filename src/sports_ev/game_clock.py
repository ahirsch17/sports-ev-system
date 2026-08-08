"""Game clock labels and ESPN live status for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from sports_ev.db.models import Game
from sports_ev.softbooks.matching import find_game_for_matchup
from sports_ev.sports.registry import get_sport_config, list_sports


def _parse_score(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class LiveGameStatus:
    home_team: str
    away_team: str
    state: str
    detail: str | None
    home_score: int | None
    away_score: int | None


def _competitor_teams(competition: dict[str, Any]) -> tuple[str | None, str | None, int | None, int | None]:
    home_team: str | None = None
    away_team: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    for competitor in competition.get("competitors") or []:
        team = (competitor.get("team") or {}).get("displayName") or (
            competitor.get("team") or {}
        ).get("name")
        if not team:
            continue
        side = str(competitor.get("homeAway", "")).lower()
        score = _parse_score(competitor.get("score"))
        if side == "home":
            home_team = str(team)
            home_score = score
        elif side == "away":
            away_team = str(team)
            away_score = score
    return home_team, away_team, home_score, away_score


def parse_espn_scoreboard_events(payload: dict[str, Any]) -> list[LiveGameStatus]:
    """Parse pre, in, and post game states from an ESPN scoreboard payload."""
    results: list[LiveGameStatus] = []
    for event in payload.get("events") or []:
        for competition in event.get("competitions") or []:
            status_type = (competition.get("status") or {}).get("type") or {}
            state = str(status_type.get("state") or "pre").lower()
            detail = status_type.get("shortDetail") or status_type.get("detail")
            home_team, away_team, home_score, away_score = _competitor_teams(competition)
            if not home_team or not away_team:
                continue
            results.append(
                LiveGameStatus(
                    home_team=home_team,
                    away_team=away_team,
                    state=state,
                    detail=str(detail) if detail else None,
                    home_score=home_score,
                    away_score=away_score,
                )
            )
    return results


def format_countdown(*, kickoff: datetime, now: datetime) -> str:
    kickoff = kickoff.astimezone(now.tzinfo or timezone.utc)
    delta = kickoff - now
    if delta.total_seconds() <= 0:
        return "Starting"

    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes < 60:
        return f"Starts in {total_minutes}m"
    hours, minutes = divmod(total_minutes, 60)
    if hours < 24:
        return f"Starts in {hours}h {minutes}m" if minutes else f"Starts in {hours}h"
    return kickoff.strftime("%a %I:%M %p").lstrip("0").replace(" 0", " ")


def format_score_line(*, away_score: int, home_score: int) -> str:
    return f"{away_score}-{home_score}"


def format_game_clock(
    *,
    kickoff: datetime,
    now: datetime,
    home_score: int | None = None,
    away_score: int | None = None,
    live: LiveGameStatus | None = None,
) -> str:
    """One-line status: countdown, live score/progress, or final."""
    if live:
        if live.state == "pre":
            return format_countdown(kickoff=kickoff, now=now)
        if live.state == "in":
            if live.away_score is not None and live.home_score is not None:
                line = format_score_line(away_score=live.away_score, home_score=live.home_score)
                progress = live.detail or "Live"
                return f"{line} · {progress}"
            return live.detail or "Live"
        if live.state == "post":
            if live.away_score is not None and live.home_score is not None:
                return f"{format_score_line(away_score=live.away_score, home_score=live.home_score)} Final"
            return "Final"

    if kickoff > now:
        return format_countdown(kickoff=kickoff, now=now)

    if home_score is not None and away_score is not None:
        line = format_score_line(away_score=away_score, home_score=home_score)
        if kickoff + timedelta(hours=5) < now:
            return f"{line} Final"
        return f"{line} · Live"

    return "In progress"


def load_live_status_by_game_id(
    session: Session,
    *,
    sports: set[str] | None = None,
) -> dict[str, LiveGameStatus]:
    """Map game_id to ESPN live status for today and yesterday slates."""
    from sports_ev.scores.espn import EspnScoreClient

    sports = sports or set(list_sports())
    client = EspnScoreClient()
    index: dict[str, LiveGameStatus] = {}
    today = date.today()
    dates = (today, today - timedelta(days=1))

    try:
        for sport in sorted(sports):
            config = get_sport_config(sport)
            for on_date in dates:
                try:
                    payload = client.fetch_scoreboard(config.espn_scoreboard_path, on_date=on_date)
                except Exception:  # noqa: BLE001
                    continue
                for status in parse_espn_scoreboard_events(payload):
                    game = find_game_for_matchup(
                        session,
                        sport=config.sport,
                        home_team=status.home_team,
                        away_team=status.away_team,
                        kickoff_time=None,
                        event_date=on_date,
                    )
                    if game is not None:
                        index[game.game_id] = status
    finally:
        client.close()

    return index


def game_clock_for(
    game: Game | None,
    *,
    now: datetime,
    live_by_game_id: dict[str, LiveGameStatus],
) -> str:
    if game is None:
        return "—"
    live = live_by_game_id.get(game.game_id)
    return format_game_clock(
        kickoff=game.kickoff_time,
        now=now,
        home_score=game.home_score,
        away_score=game.away_score,
        live=live,
    )
