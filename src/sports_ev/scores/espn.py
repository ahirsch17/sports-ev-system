from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sports_ev.softbooks.base import RateLimitedClient, SoftBookScraperError


@dataclass(frozen=True)
class ParsedGameScore:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    completed: bool
    event_date: date | None = None


ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"


def _parse_score(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_espn_scoreboard(payload: dict[str, Any]) -> list[ParsedGameScore]:
    results: list[ParsedGameScore] = []
    for event in payload.get("events") or []:
        event_date: date | None = None
        raw_date = event.get("date")
        if raw_date:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            event_date = dt.date()

        for competition in event.get("competitions") or []:
            status = (competition.get("status") or {}).get("type") or {}
            completed = bool(status.get("completed")) or status.get("state") == "post"
            if not completed:
                continue

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

            if (
                home_team
                and away_team
                and home_score is not None
                and away_score is not None
            ):
                results.append(
                    ParsedGameScore(
                        home_team=home_team,
                        away_team=away_team,
                        home_score=home_score,
                        away_score=away_score,
                        completed=True,
                        event_date=event_date,
                    )
                )

    return results


class EspnScoreClient:
    def __init__(self, *, http: RateLimitedClient | None = None):
        self.http = http or RateLimitedClient(min_interval_seconds=1.0)

    def close(self) -> None:
        self.http.close()

    def fetch_scoreboard(self, espn_path: str, *, on_date: date | None = None) -> dict[str, Any]:
        url = f"{ESPN_BASE}/{espn_path.strip('/')}"
        params = {"dates": on_date.strftime("%Y%m%d")} if on_date else None
        return self.http.get_json(url, params=params)

    def fetch_completed_scores(
        self, espn_path: str, *, on_date: date | None = None
    ) -> list[ParsedGameScore]:
        try:
            payload = self.fetch_scoreboard(espn_path, on_date=on_date)
        except SoftBookScraperError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SoftBookScraperError(f"ESPN scoreboard fetch failed: {exc}") from exc
        return parse_espn_scoreboard(payload)
