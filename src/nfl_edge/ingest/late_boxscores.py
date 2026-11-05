"""Final box scores the weekly stat feed has not published yet.

The schedule gets the score first. Passing and receiving yards are already on
the ESPN summary for that game's ``espn`` id. Until the stat feed has the
``game_id``, those lines are read from there and added. Once the feed has the
game, they are skipped.
"""

from __future__ import annotations

import json
import logging
from urllib.request import urlopen

import polars as pl

from nfl_edge.config import CACHE_DIR, ensure_dirs

log = logging.getLogger(__name__)

_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={espn}"


def _num(text: object) -> float:
    try:
        return float(str(text).split("-")[0].split("/")[0])
    except (TypeError, ValueError):
        return 0.0


def _summary(espn_id: str) -> dict | None:
    ensure_dirs()
    path = CACHE_DIR / f"espn_summary_{espn_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        with urlopen(_SUMMARY.format(espn=espn_id), timeout=20) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:
        log.warning("Could not read ESPN summary %s (%s)", espn_id, exc)
        return None
    path.write_text(json.dumps(payload))
    return payload


def parse_summary(payload: dict, game: dict) -> tuple[list[dict], list[dict]]:
    """Player lines and one row per team.

    Team passing yards are the quarterback's gross yards. That is what the
    weekly team file stores. The box score's own team passing total is net of
    sacks, so it is not the number used here.
    """
    competitors = (payload.get("header") or {}).get("competitions", [{}])[0].get("competitors") or []
    abbr_to_team: dict[str, str] = {}
    for side in competitors:
        abbr = str((side.get("team") or {}).get("abbreviation") or "")
        if side.get("homeAway") == "home":
            abbr_to_team[abbr] = str(game["home_team"])
        elif side.get("homeAway") == "away":
            abbr_to_team[abbr] = str(game["away_team"])

    players: dict[str, dict] = {}
    for team_block in (payload.get("boxscore") or {}).get("players") or []:
        abbr = str((team_block.get("team") or {}).get("abbreviation") or "")
        team = abbr_to_team.get(abbr)
        if not team:
            continue
        opponent = str(game["away_team"] if team == game["home_team"] else game["home_team"])
        for group in team_block.get("statistics") or []:
            kind = group.get("name")
            if kind not in {"passing", "rushing", "receiving"}:
                continue
            labels = list(group.get("labels") or [])
            for athlete in group.get("athletes") or []:
                name = str((athlete.get("athlete") or {}).get("displayName") or "")
                if not name:
                    continue
                values = dict(zip(labels, athlete.get("stats") or []))
                slot = players.setdefault(
                    name,
                    {"player_display_name": name, "team": team, "opponent_team": opponent},
                )
                if kind == "passing":
                    made, att = (str(values.get("C/ATT") or "0/0") + "/0").split("/")[:2]
                    slot["completions"] = int(_num(made))
                    slot["attempts"] = int(_num(att))
                    slot["passing_yards"] = _num(values.get("YDS"))
                    slot["passing_tds"] = _num(values.get("TD"))
                    slot["passing_interceptions"] = _num(values.get("INT"))
                    sacks = str(values.get("SACKS") or "0-0").split("-")
                    slot["sacks_suffered"] = _num(sacks[0])
                    slot["sack_yards_lost"] = -_num(sacks[1] if len(sacks) > 1 else 0)
                elif kind == "rushing":
                    slot["carries"] = _num(values.get("CAR"))
                    slot["rushing_yards"] = _num(values.get("YDS"))
                    slot["rushing_tds"] = _num(values.get("TD"))
                else:
                    slot["receptions"] = _num(values.get("REC"))
                    slot["receiving_yards"] = _num(values.get("YDS"))
                    slot["receiving_tds"] = _num(values.get("TD"))
                    slot["targets"] = _num(values.get("TGTS"))

    team_rows: dict[str, dict] = {}
    counted = (
        "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
        "sacks_suffered", "sack_yards_lost", "carries", "rushing_yards", "rushing_tds",
    )
    for slot in players.values():
        row = team_rows.setdefault(
            slot["team"],
            {"team": slot["team"], "opponent_team": slot["opponent_team"], **{key: 0.0 for key in counted}},
        )
        for key in counted:
            row[key] += float(slot.get(key) or 0)
    return list(players.values()), list(team_rows.values())


def week_should_fill(
    present: int,
    week_size: int,
    *,
    season: int,
    earlier_counts: list[int],
    current_season: int,
) -> bool:
    """Fill a finished week the stat file skipped.

    A week that is already mostly in the file gets its missing games. A week
    the file has not started yet is filled only when this frame is the real
    current season — an earlier week already has a full slate. A one-game
    test frame, and any older season, is left alone.
    """
    if week_size <= 0 or present >= week_size:
        return False
    if present * 2 >= week_size:
        return True
    if season != current_season:
        return False
    return any(count >= 8 for count in earlier_counts)


def _earlier_counts(stats: pl.DataFrame, season: int, week: int) -> list[int]:
    if "game_id" not in stats.columns or "week" not in stats.columns:
        return []
    prior = stats.filter((pl.col("season") == season) & (pl.col("week") < week))
    if prior.is_empty():
        return []
    return [int(n) for n in prior.group_by("week").agg(pl.col("game_id").n_unique())["game_id"].to_list()]


def _missing_games(stats: pl.DataFrame) -> list[dict]:
    from nfl_edge.config import current_nfl_season
    from nfl_edge.ingest.nflverse import load_schedules

    seasons = [int(s) for s in stats["season"].unique().to_list() if s is not None]
    if not seasons:
        return []
    schedules = load_schedules(seasons)
    need = {"game_id", "espn", "home_team", "away_team", "home_score", "season", "week"}
    if schedules.is_empty() or not need.issubset(schedules.columns):
        return []
    have = set(stats["game_id"].drop_nulls().to_list()) if "game_id" in stats.columns else set()
    final = schedules.filter(pl.col("home_score").is_not_null() & pl.col("espn").is_not_null())
    current = current_nfl_season()
    holes: list[dict] = []
    for keys, week_games in final.group_by(["season", "week"]):
        del keys
        ids = week_games["game_id"].to_list()
        present = sum(1 for game_id in ids if game_id in have)
        season = int(week_games["season"][0])
        week = int(week_games["week"][0])
        if not week_should_fill(
            present,
            len(ids),
            season=season,
            earlier_counts=_earlier_counts(stats, season, week),
            current_season=current,
        ):
            continue
        missing = week_games.filter(~pl.col("game_id").is_in(list(have)))
        holes.extend(missing.select(["game_id", "espn", "home_team", "away_team", "season", "week"]).iter_rows(named=True))
    return holes


def _box_is_usable(players: list[dict]) -> bool:
    """A renamed or empty box score is not a real game. Leave it missing."""
    return any(float(row.get("attempts") or 0) > 0 or float(row.get("passing_yards") or 0) > 0 for row in players)


def finals_missing(stats: pl.DataFrame, schedules: pl.DataFrame) -> list[dict]:
    """Finished games that still have no player lines after the box-score fill."""
    need = {"game_id", "home_team", "away_team", "home_score", "season", "week"}
    if stats.is_empty() or schedules.is_empty() or not need.issubset(schedules.columns):
        return []
    final = schedules.filter(pl.col("home_score").is_not_null())
    if "game_type" in final.columns:
        final = final.filter(pl.col("game_type") == "REG")
    have = set(stats["game_id"].drop_nulls().to_list()) if "game_id" in stats.columns else set()
    missing = final.filter(~pl.col("game_id").is_in(list(have)))
    if missing.is_empty():
        return []
    rows = missing.select(["game_id", "season", "week", "away_team", "home_team"]).sort(["week", "game_id"])
    return rows.to_dicts()


def _zeroed(template: pl.DataFrame, known: dict) -> pl.DataFrame:
    row = template.head(1)
    numeric = [
        pl.lit(0, dtype=dtype).alias(name)
        for name, dtype in row.schema.items()
        if dtype.is_numeric() and name not in {"season", "week"}
    ]
    if numeric:
        row = row.with_columns(numeric)
    sets = []
    for key, value in known.items():
        if key not in row.columns or value is None:
            continue
        sets.append(pl.lit(value, dtype=row.schema[key]).alias(key))
    return row.with_columns(sets) if sets else row


def _infer_position(slot: dict) -> str:
    if float(slot.get("attempts") or 0) >= 8:
        return "QB"
    if float(slot.get("carries") or 0) > float(slot.get("targets") or 0):
        return "RB"
    return "WR"


def _games_to_add(stats: pl.DataFrame) -> list[tuple[dict, list[dict], list[dict]]]:
    if stats.is_empty() or "game_id" not in stats.columns or "season" not in stats.columns:
        return []
    found = []
    for game in _missing_games(stats):
        payload = _summary(str(game["espn"]))
        if not payload:
            continue
        players, teams = parse_summary(payload, game)
        if players and _box_is_usable(players):
            found.append((game, players, teams))
    return found


def apply_player_boxscores(stats: pl.DataFrame) -> pl.DataFrame:
    if stats.is_empty() or "player_display_name" not in stats.columns:
        return stats
    pieces: list[pl.DataFrame] = []
    fallback = stats.head(1)
    for game, players, _teams in _games_to_add(stats):
        for slot in players:
            name = slot["player_display_name"]
            own = stats.filter(pl.col("player_display_name") == name)
            template = own.sort("season", "week").tail(1) if own.height else fallback
            known = {
                "season": int(game["season"]),
                "week": int(game["week"]),
                "season_type": "REG",
                "game_id": game["game_id"],
                **slot,
            }
            if not own.height:
                known["position"] = _infer_position(slot)
                known["player_name"] = name
            pieces.append(_zeroed(template, known))
    if not pieces:
        return stats
    return pl.concat([stats, *pieces], how="diagonal_relaxed")


def apply_team_boxscores(stats: pl.DataFrame) -> pl.DataFrame:
    if stats.is_empty() or "team" not in stats.columns:
        return stats
    pieces: list[pl.DataFrame] = []
    for game, _players, teams in _games_to_add(stats):
        for slot in teams:
            own = stats.filter(pl.col("team") == slot["team"])
            template = own.sort("season", "week").tail(1) if own.height else stats.head(1)
            known = {
                "season": int(game["season"]),
                "week": int(game["week"]),
                "season_type": "REG",
                "game_id": game["game_id"],
                **slot,
            }
            pieces.append(_zeroed(template, known))
    if not pieces:
        return stats
    return pl.concat([stats, *pieces], how="diagonal_relaxed")
