from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.softbooks.base import RateLimitedClient
from sports_ev.sports.registry import SportConfig, get_sport_config


@dataclass
class FixtureSyncResult:
    sport: str = ""
    matchups_seen: int = 0
    games_upserted: int = 0
    errors: list[str] = field(default_factory=list)


def _season_from_kickoff(sport: str, kickoff: datetime) -> int:
    """Season label for fixture rows."""
    kickoff = kickoff.astimezone(timezone.utc)
    if sport == "nfl" and kickoff.month <= 2:
        return kickoff.year - 1
    return kickoff.year


class PinnacleFixtureSync:
    """Upsert games from Pinnacle guest API matchups — no API key required."""

    BASE_URL = "https://guest.api.arcadia.pinnacle.com/0.1"

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.http = RateLimitedClient(min_interval_seconds=2.0)

    def close(self) -> None:
        self.http.close()

    def fetch_matchups(self, league_id: int) -> list[dict]:
        url = f"{self.BASE_URL}/leagues/{league_id}/matchups"
        return self.http.get_json(url)

    def sync_fixtures(self, sport: str) -> FixtureSyncResult:
        config = get_sport_config(sport, self.settings)
        result = FixtureSyncResult(sport=config.sport)
        try:
            matchups = self.fetch_matchups(config.pinnacle_league_id)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))
            return result

        result.matchups_seen = len(matchups)
        for matchup in matchups:
            if matchup.get("type") != "matchup" or matchup.get("isLive"):
                continue
            participants = {
                p.get("alignment"): p.get("name")
                for p in (matchup.get("participants") or [])
                if p.get("alignment") and p.get("name")
            }
            home = participants.get("home")
            away = participants.get("away")
            if not home or not away:
                continue

            start = matchup.get("startTime")
            if not start:
                continue
            kickoff = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)

            game_id = config.game_id_from_matchup(int(matchup["id"]))
            game = self.session.query(Game).filter_by(game_id=game_id).one_or_none()
            if game is None:
                game = Game(
                    game_id=game_id,
                    sport=config.sport,
                    home_team=str(home),
                    away_team=str(away),
                    kickoff_time=kickoff,
                    season=_season_from_kickoff(config.sport, kickoff),
                )
                self.session.add(game)
                result.games_upserted += 1
            else:
                game.home_team = str(home)
                game.away_team = str(away)
                game.kickoff_time = kickoff
                game.season = _season_from_kickoff(config.sport, kickoff)

        self.session.flush()
        return result

    def sync_nfl_fixtures(self) -> FixtureSyncResult:
        """Backward-compatible NFL sync."""
        return self.sync_fixtures("nfl")

    def sync_all(self) -> list[FixtureSyncResult]:
        from sports_ev.sports.registry import list_sports

        return [self.sync_fixtures(sport) for sport in list_sports()]
