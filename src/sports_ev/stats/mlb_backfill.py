from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from sports_ev.db.models import Game, TeamGameStat
from sports_ev.softbooks.base import RateLimitedClient, SoftBookScraperError
from sports_ev.softbooks.matching import teams_match


MLB_STATS_API = "https://statsapi.mlb.com/api/v1"


@dataclass
class MlbStatsBackfillResult:
    games_checked: int = 0
    games_matched: int = 0
    stats_inserted: int = 0
    errors: list[str] = field(default_factory=list)


def _parse_game_date(raw: str) -> date:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.date()


def _pitching_totals(boxscore: dict[str, Any], side: str) -> tuple[float, float, float]:
    teams = boxscore.get("teams") or {}
    team = teams.get(side) or {}
    players = team.get("players") or {}
    starter_innings = 0.0
    starter_er = 0.0
    bullpen_er = 0.0
    found_starter = False

    for player in players.values():
        person = player.get("person") or {}
        position = (person.get("primaryPosition") or {}).get("abbreviation")
        stats = (player.get("stats") or {}).get("pitching") or {}
        if not stats:
            continue
        innings = _parse_innings(stats.get("inningsPitched", "0"))
        er = float(stats.get("earnedRuns") or 0)
        if position == "P" and not found_starter and innings > 0:
            starter_innings = innings
            starter_er = er
            found_starter = True
        elif innings > 0:
            bullpen_er += er

    return starter_innings, starter_er, bullpen_er


def _parse_innings(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value)
    if "." in text:
        whole, frac = text.split(".", 1)
        outs = int(frac[:1] or "0")
        return float(whole) + outs / 3.0
    return float(text)


def _team_box_hits(boxscore: dict[str, Any], side: str) -> float:
    team = ((boxscore.get("teams") or {}).get(side) or {}).get("teamStatistics") or {}
    batting = team.get("batting") or {}
    return float(batting.get("hits") or 0)


def extract_team_stats_from_boxscore(
    boxscore: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    known_at: datetime,
) -> list[TeamGameStat]:
    rows: list[TeamGameStat] = []

    home_starter_inn, home_starter_er, home_bullpen_er = _pitching_totals(boxscore, "home")
    away_starter_inn, away_starter_er, away_bullpen_er = _pitching_totals(boxscore, "away")

    common = [
        (home_team, float(home_score), float(away_score), home_starter_inn, home_starter_er, home_bullpen_er),
        (away_team, float(away_score), float(home_score), away_starter_inn, away_starter_er, away_bullpen_er),
    ]
    for team, rs, ra, si, ser, ber in common:
        side = "home" if team == home_team else "away"
        hits = _team_box_hits(boxscore, side)
        stat_rows = [
            ("runs_scored", rs),
            ("runs_allowed", ra),
            ("hits", hits),
            ("starter_innings", si),
            ("starter_earned_runs", ser),
            ("bullpen_earned_runs", ber),
        ]
        for key, value in stat_rows:
            rows.append(
                TeamGameStat(
                    game_id="",
                    team=team,
                    stat_key=key,
                    stat_value=float(value),
                    known_at=known_at,
                )
            )
    return rows


class MlbStatsBackfillService:
    """Backfill team_game_stats for settled MLB games using the public MLB Stats API."""

    def __init__(self, session: Session, *, http: RateLimitedClient | None = None):
        self.session = session
        self.http = http or RateLimitedClient(min_interval_seconds=1.0)
        self._owns_http = http is None

    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    def fetch_schedule(self, start: date, end: date) -> list[dict[str, Any]]:
        url = f"{MLB_STATS_API}/schedule"
        params = {
            "sportId": 1,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "gameType": "R",
        }
        payload = self.http.get_json(url, params=params)
        games: list[dict[str, Any]] = []
        for day in payload.get("dates") or []:
            games.extend(day.get("games") or [])
        return games

    def fetch_boxscore(self, game_pk: int) -> dict[str, Any]:
        return self.http.get_json(f"{MLB_STATS_API}/game/{game_pk}/boxscore")

    def _match_db_game(self, games: list[Game], api_game: dict[str, Any]) -> Game | None:
        teams = api_game.get("teams") or {}
        home_name = ((teams.get("home") or {}).get("team") or {}).get("name", "")
        away_name = ((teams.get("away") or {}).get("team") or {}).get("name", "")
        game_date = _parse_game_date(api_game.get("gameDate", ""))

        matches: list[Game] = []
        for game in games:
            if teams_match(game.home_team, home_name) and teams_match(game.away_team, away_name):
                if game.kickoff_time.astimezone(timezone.utc).date() == game_date:
                    matches.append(game)
        if len(matches) == 1:
            return matches[0]
        if matches:
            return matches[0]
        return None

    def backfill(
        self,
        *,
        lookback_days: int = 120,
        only_missing: bool = True,
        fetch_boxscores: bool = True,
    ) -> MlbStatsBackfillResult:
        result = MlbStatsBackfillResult()
        today = date.today()
        start = today - timedelta(days=lookback_days)

        db_games = (
            self.session.query(Game)
            .filter(
                Game.sport == "mlb",
                Game.home_score.isnot(None),
                Game.away_score.isnot(None),
                Game.kickoff_time >= datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            )
            .all()
        )
        if only_missing:
            existing = {
                row[0]
                for row in self.session.query(TeamGameStat.game_id).join(Game).filter(Game.sport == "mlb").distinct()
            }
            db_games = [g for g in db_games if g.game_id not in existing]

        result.games_checked = len(db_games)
        if not db_games:
            return result

        if not fetch_boxscores:
            for db_game in db_games:
                known_at = db_game.kickoff_time + timedelta(hours=4)
                stat_rows = extract_team_stats_from_boxscore(
                    {"teams": {"home": {}, "away": {}}},
                    home_team=db_game.home_team,
                    away_team=db_game.away_team,
                    home_score=int(db_game.home_score or 0),
                    away_score=int(db_game.away_score or 0),
                    known_at=known_at,
                )
                for row in stat_rows:
                    row.game_id = db_game.game_id
                    self.session.add(row)
                    result.stats_inserted += 1
                result.games_matched += 1
            self.session.flush()
            return result

        try:
            api_games = self.fetch_schedule(start, today)
        except SoftBookScraperError as exc:
            result.errors.append(str(exc))
            return result

        final_api_games = [
            g
            for g in api_games
            if ((g.get("status") or {}).get("abstractGameState") == "Final")
        ]

        for api_game in final_api_games:
            db_game = self._match_db_game(db_games, api_game)
            if db_game is None:
                continue

            game_pk = api_game.get("gamePk")
            if game_pk is None:
                continue

            try:
                if fetch_boxscores:
                    boxscore = self.fetch_boxscore(int(game_pk))
                else:
                    boxscore = {"teams": {"home": {}, "away": {}}}
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"boxscore {game_pk}: {exc}")
                continue

            known_at = db_game.kickoff_time + timedelta(hours=4)
            stat_rows = extract_team_stats_from_boxscore(
                boxscore,
                home_team=db_game.home_team,
                away_team=db_game.away_team,
                home_score=int(db_game.home_score or 0),
                away_score=int(db_game.away_score or 0),
                known_at=known_at,
            )
            for row in stat_rows:
                row.game_id = db_game.game_id
                self.session.add(row)
                result.stats_inserted += 1
            result.games_matched += 1

        self.session.flush()
        return result

    def backfill_from_schedule(
        self,
        *,
        lookback_days: int = 120,
        fetch_boxscores: bool = False,
    ) -> MlbStatsBackfillResult:
        """Upsert final scores, insert missing games, then backfill team stats."""
        result = MlbStatsBackfillResult()
        today = date.today()
        start = today - timedelta(days=lookback_days)

        db_games = self.session.query(Game).filter(Game.sport == "mlb").all()
        pending_ids = {g.game_id for g in db_games}

        try:
            api_games = self.fetch_schedule(start, today + timedelta(days=1))
        except SoftBookScraperError as exc:
            result.errors.append(str(exc))
            return result

        for api_game in api_games:
            status = (api_game.get("status") or {}).get("abstractGameState")
            if status != "Final":
                continue

            teams = api_game.get("teams") or {}
            home_name = ((teams.get("home") or {}).get("team") or {}).get("name", "")
            away_name = ((teams.get("away") or {}).get("team") or {}).get("name", "")
            home_score = (teams.get("home") or {}).get("score")
            away_score = (teams.get("away") or {}).get("score")
            if not home_name or not away_name or home_score is None or away_score is None:
                continue

            kickoff = datetime.fromisoformat(api_game["gameDate"].replace("Z", "+00:00"))
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            season = kickoff.year
            game_pk = api_game.get("gamePk")
            game_id = f"mlb_api_{game_pk}" if game_pk else f"mlb_api_{kickoff.date()}_{home_name}"

            db_game = self._match_db_game(db_games, api_game)
            if db_game is None and game_id not in pending_ids:
                db_game = self.session.query(Game).filter_by(game_id=game_id).one_or_none()
            if db_game is None and game_id not in pending_ids:
                db_game = Game(
                    game_id=game_id,
                    sport="mlb",
                    home_team=home_name,
                    away_team=away_name,
                    kickoff_time=kickoff,
                    season=season,
                    home_score=int(home_score),
                    away_score=int(away_score),
                )
                self.session.add(db_game)
                db_games.append(db_game)
                pending_ids.add(game_id)
                self.session.flush()
            elif db_game is not None:
                db_game.home_score = int(home_score)
                db_game.away_score = int(away_score)
                pending_ids.add(db_game.game_id)
                if db_game not in db_games:
                    db_games.append(db_game)

            if db_game is not None and db_game in self.session.new:
                self.session.flush()

        self.session.flush()
        stats_result = self.backfill(
            lookback_days=lookback_days,
            only_missing=True,
            fetch_boxscores=fetch_boxscores,
        )
        result.games_checked = stats_result.games_checked
        result.games_matched = stats_result.games_matched
        result.stats_inserted = stats_result.stats_inserted
        result.errors.extend(stats_result.errors)
        return result
