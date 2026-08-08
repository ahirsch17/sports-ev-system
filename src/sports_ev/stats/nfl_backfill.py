from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from sqlalchemy.orm import Session

from sports_ev.db.models import Game, OddsSnapshot, TeamGameStat

GAMES_CSV = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
TEAMS_CSV = "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv"
WEEK_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
PARSER_VERSION = "nflverse@1.0"


@dataclass
class NflBackfillResult:
    games_seen: int = 0
    games_upserted: int = 0
    odds_inserted: int = 0
    stats_inserted: int = 0
    errors: list[str] = field(default_factory=list)


def _fetch_text(url: str) -> str:
    response = httpx.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    return response.text


def _team_name_map() -> dict[str, str]:
    text = _fetch_text(TEAMS_CSV)
    return {
        row["team_abbr"]: row["team_name"]
        for row in csv.DictReader(io.StringIO(text))
        if row.get("team_abbr") and row.get("team_name")
    }


def _parse_kickoff(row: dict[str, str]) -> datetime:
    gameday = row.get("gameday") or ""
    gametime = (row.get("gametime") or "13:00").strip()
    if not gameday:
        raise ValueError("missing gameday")
    dt = datetime.fromisoformat(f"{gameday}T{gametime}")
    return dt.replace(tzinfo=timezone.utc)


def _home_spread_line(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    spread = float(raw)
    # nflverse: positive spread_line => home favored => home gives points.
    return -abs(spread) if spread > 0 else spread


def _game_key(nflverse_game_id: str) -> str:
    return f"nfl_{nflverse_game_id}"


class NflHistoryBackfillService:
    """Load completed NFL games, spreads, and weekly EPA stats from nflverse."""

    def __init__(self, session: Session):
        self.session = session

    def backfill_seasons(self, seasons: list[int]) -> NflBackfillResult:
        result = NflBackfillResult()
        try:
            team_map = _team_name_map()
            games_text = _fetch_text(GAMES_CSV)
            rows = list(csv.DictReader(io.StringIO(games_text)))
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))
            return result

        season_set = {str(s) for s in seasons}
        for row in rows:
            if row.get("season") not in season_set:
                continue
            if row.get("game_type") not in {"REG", "POST"}:
                continue
            if not row.get("home_score") or not row.get("away_score"):
                continue

            result.games_seen += 1
            try:
                home_abbr = row["home_team"]
                away_abbr = row["away_team"]
                home_team = team_map.get(home_abbr, home_abbr)
                away_team = team_map.get(away_abbr, away_abbr)
                kickoff = _parse_kickoff(row)
                game_id = _game_key(row["game_id"])
                spread = _home_spread_line(row.get("spread_line"))

                game = self.session.query(Game).filter_by(game_id=game_id).one_or_none()
                if game is None:
                    game = Game(
                        game_id=game_id,
                        sport="nfl",
                        home_team=home_team,
                        away_team=away_team,
                        kickoff_time=kickoff,
                        season=int(row["season"]),
                        week=int(row["week"]) if row.get("week") else None,
                    )
                    self.session.add(game)
                    result.games_upserted += 1

                game.home_score = int(float(row["home_score"]))
                game.away_score = int(float(row["away_score"]))

                if spread is not None:
                    captured = kickoff.replace(hour=12, minute=0, second=0, microsecond=0)
                    for book in ("pinnacle", "draftkings"):
                        exists = (
                            self.session.query(OddsSnapshot)
                            .filter_by(
                                game_id=game_id,
                                book=book,
                                market_type="spread",
                            )
                            .first()
                        )
                        if exists is None:
                            self.session.add(
                                OddsSnapshot(
                                    game_id=game_id,
                                    book=book,
                                    market_type="spread",
                                    line=spread,
                                    odds_home=-110,
                                    odds_away=-110,
                                    captured_at=captured,
                                    source="backfill",
                                    parser_version=PARSER_VERSION,
                                )
                            )
                            result.odds_inserted += 1
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{row.get('game_id')}: {exc}")

        self.session.flush()

        for season in seasons:
            try:
                result.stats_inserted += self._backfill_week_stats(season, team_map)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"stats {season}: {exc}")

        self.session.flush()
        return result

    def _backfill_week_stats(self, season: int, team_map: dict[str, str]) -> int:
        url = WEEK_STATS_URL.format(season=season)
        text = _fetch_text(url)
        inserted = 0
        for row in csv.DictReader(io.StringIO(text)):
            nflverse_id = row.get("game_id")
            team_abbr = row.get("team")
            if not nflverse_id or not team_abbr:
                continue
            game_id = _game_key(nflverse_id)
            if self.session.query(Game).filter_by(game_id=game_id).one_or_none() is None:
                continue

            team = team_map.get(team_abbr, team_abbr)
            passing_epa = _float_or_none(row.get("passing_epa"))
            rushing_epa = _float_or_none(row.get("rushing_epa"))
            if passing_epa is None and rushing_epa is None:
                continue

            epa_offense = (passing_epa or 0.0) + (rushing_epa or 0.0)
            game = self.session.query(Game).filter_by(game_id=game_id).one()
            known_at = game.kickoff_time

            stat_rows = [
                ("epa_offense", epa_offense),
                ("passing_epa", passing_epa),
                ("rushing_epa", rushing_epa),
                ("success_rate", _float_or_none(row.get("success_rate"))),
            ]
            for stat_key, stat_value in stat_rows:
                if stat_value is None:
                    continue
                exists = (
                    self.session.query(TeamGameStat)
                    .filter_by(game_id=game_id, team=team, stat_key=stat_key)
                    .first()
                )
                if exists is not None:
                    continue
                self.session.add(
                    TeamGameStat(
                        game_id=game_id,
                        team=team,
                        stat_key=stat_key,
                        stat_value=stat_value,
                        known_at=known_at,
                    )
                )
                inserted += 1
        return inserted


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None
