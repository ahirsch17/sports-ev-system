from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from sports_ev.db.models import Game, TeamGameStat
from sports_ev.features.mlb.pitcher import PROBABLE_STAT_KEYS
from sports_ev.softbooks.base import RateLimitedClient, SoftBookScraperError
from sports_ev.softbooks.matching import teams_match

MLB_STATS_API = "https://statsapi.mlb.com/api/v1"


@dataclass
class MlbProbableSyncResult:
    games_checked: int = 0
    games_updated: int = 0
    stats_inserted: int = 0
    errors: list[str] = field(default_factory=list)


def _pitching_rates(stats_block: dict[str, Any]) -> dict[str, float]:
    splits = (stats_block.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return {}
    stat = (splits[0].get("stat") or {})
    era = stat.get("era")
    whip = stat.get("whip")
    innings = stat.get("inningsPitched", "0")
    strikeouts = float(stat.get("strikeOuts") or 0)
    ip = _parse_innings(innings)
    k9 = (strikeouts / ip * 9.0) if ip > 0 else None

    result: dict[str, float] = {}
    if era is not None:
        result["probable_starter_era"] = float(era)
    if whip is not None:
        result["probable_starter_whip"] = float(whip)
    if k9 is not None:
        result["probable_starter_k9"] = float(k9)
    return result


def _parse_innings(value: Any) -> float:
    text = str(value)
    if "." in text:
        whole, frac = text.split(".", 1)
        outs = int(frac[:1] or "0")
        return float(whole) + outs / 3.0
    return float(text or 0)


class MlbProbableSyncService:
    """Sync probable starter season stats for upcoming MLB games."""

    def __init__(self, session: Session, *, http: RateLimitedClient | None = None):
        self.session = session
        self.http = http or RateLimitedClient(min_interval_seconds=1.0)
        self._owns_http = http is None

    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    def fetch_schedule(self, on_date: date) -> list[dict[str, Any]]:
        payload = self.http.get_json(
            f"{MLB_STATS_API}/schedule",
            params={"sportId": 1, "date": on_date.strftime("%Y-%m-%d"), "gameType": "R"},
        )
        games: list[dict[str, Any]] = []
        for day in payload.get("dates") or []:
            games.extend(day.get("games") or [])
        return games

    def fetch_pitcher_stats(self, pitcher_id: int, *, season: int) -> dict[str, float]:
        payload = self.http.get_json(
            f"{MLB_STATS_API}/people/{pitcher_id}/stats",
            params={"stats": "season", "season": season, "group": "pitching"},
        )
        return _pitching_rates(payload)

    def _match_game(self, games: list[Game], api_game: dict[str, Any]) -> Game | None:
        teams = api_game.get("teams") or {}
        home_name = ((teams.get("home") or {}).get("team") or {}).get("name", "")
        away_name = ((teams.get("away") or {}).get("team") or {}).get("name", "")
        kickoff = datetime.fromisoformat(api_game["gameDate"].replace("Z", "+00:00"))
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        target_date = kickoff.date()

        for game in games:
            if teams_match(game.home_team, home_name) and teams_match(game.away_team, away_name):
                if game.kickoff_time.astimezone(timezone.utc).date() == target_date:
                    return game
        return None

    def _existing_probables(self, game_id: str) -> bool:
        return (
            self.session.query(TeamGameStat)
            .filter(
                TeamGameStat.game_id == game_id,
                TeamGameStat.stat_key.in_(PROBABLE_STAT_KEYS),
            )
            .count()
            > 0
        )

    def sync_upcoming(self, *, days_ahead: int = 7) -> MlbProbableSyncResult:
        result = MlbProbableSyncResult()
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days_ahead)

        db_games = (
            self.session.query(Game)
            .filter(Game.sport == "mlb", Game.kickoff_time > now, Game.kickoff_time <= end)
            .all()
        )
        result.games_checked = len(db_games)
        if not db_games:
            return result

        today = date.today()
        dates = [today + timedelta(days=i) for i in range(days_ahead + 1)]

        try:
            api_games: list[dict[str, Any]] = []
            for on_date in dates:
                api_games.extend(self.fetch_schedule(on_date))
        except SoftBookScraperError as exc:
            result.errors.append(str(exc))
            return result

        known_at = datetime.now(timezone.utc)

        for api_game in api_games:
            db_game = self._match_game(db_games, api_game)
            if db_game is None:
                continue
            if self._existing_probables(db_game.game_id):
                continue

            teams = api_game.get("teams") or {}
            season = db_game.season
            updated = False

            for side, alignment in (("home", db_game.home_team), ("away", db_game.away_team)):
                team_block = teams.get("home" if side == "home" else "away") or {}
                probable = team_block.get("probablePitcher") or {}
                pitcher_id = probable.get("id")
                if not pitcher_id:
                    continue
                try:
                    rates = self.fetch_pitcher_stats(int(pitcher_id), season=season)
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"pitcher {pitcher_id}: {exc}")
                    continue

                for key, value in rates.items():
                    self.session.add(
                        TeamGameStat(
                            game_id=db_game.game_id,
                            team=alignment,
                            stat_key=key,
                            stat_value=value,
                            known_at=known_at,
                        )
                    )
                    result.stats_inserted += 1
                    updated = True

            if updated:
                result.games_updated += 1

        self.session.flush()
        return result

    def backfill_from_schedule(
        self,
        *,
        lookback_days: int = 120,
    ) -> MlbProbableSyncResult:
        """Attach probable starter stats from historical schedule entries when available."""
        result = MlbProbableSyncResult()
        today = date.today()
        start = today - timedelta(days=lookback_days)

        db_games = (
            self.session.query(Game)
            .filter(
                Game.sport == "mlb",
                Game.home_score.isnot(None),
                Game.kickoff_time >= datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            )
            .all()
        )
        result.games_checked = len(db_games)

        dates = [start + timedelta(days=i) for i in range((today - start).days + 1)]
        try:
            api_games: list[dict[str, Any]] = []
            for on_date in dates:
                api_games.extend(self.fetch_schedule(on_date))
        except SoftBookScraperError as exc:
            result.errors.append(str(exc))
            return result

        for api_game in api_games:
            db_game = self._match_game(db_games, api_game)
            if db_game is None or self._existing_probables(db_game.game_id):
                continue

            teams = api_game.get("teams") or {}
            known_at = db_game.kickoff_time - timedelta(hours=2)
            updated = False

            for side_key, team_name in (("home", db_game.home_team), ("away", db_game.away_team)):
                probable = (teams.get(side_key) or {}).get("probablePitcher") or {}
                pitcher_id = probable.get("id")
                if not pitcher_id:
                    continue
                try:
                    rates = self.fetch_pitcher_stats(int(pitcher_id), season=db_game.season)
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"pitcher {pitcher_id}: {exc}")
                    continue

                for key, value in rates.items():
                    self.session.add(
                        TeamGameStat(
                            game_id=db_game.game_id,
                            team=team_name,
                            stat_key=key,
                            stat_value=value,
                            known_at=known_at,
                        )
                    )
                    result.stats_inserted += 1
                    updated = True

            if updated:
                result.games_updated += 1

        self.session.flush()
        return result
