"""MLB Stats API ingest (no key required)."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl
import requests

from mlb_edge.config import CACHE_DIR, current_mlb_season, ensure_dirs

BASE = "https://statsapi.mlb.com/api/v1"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-edge/0.1 (local research)"})


def _get(path: str, params: dict | None = None) -> Any:
    url = f"{BASE}{path}"
    resp = SESSION.get(url, params=params or {}, timeout=45)
    resp.raise_for_status()
    return resp.json()


LIVE_CACHE_TTL_HOURS = float(os.getenv("MLB_EDGE_CACHE_TTL_HOURS", "2"))


def _cache_read(name: str, *, max_age_hours: float | None = None) -> Any | None:
    path = CACHE_DIR / name
    if not path.exists():
        return None
    if max_age_hours is not None and (time.time() - path.stat().st_mtime) / 3600 >= max_age_hours:
        return None
    return json.loads(path.read_text())


def _live_ttl(end: date) -> float | None:
    """Ranges touching yesterday or later still change (scores, probables); past ranges are final."""
    return LIVE_CACHE_TTL_HOURS if end >= date.today() - timedelta(days=1) else None


def _season_ttl(season: int) -> float | None:
    """Season-to-date stats for the live season move daily; past seasons are final."""
    return 12.0 if season >= current_mlb_season() else None


def refresh_live() -> dict:
    """Drop caches that can still change (recent schedule ranges, this season's stats)."""
    ensure_dirs()
    season = current_mlb_season()
    removed = 0
    for path in CACHE_DIR.glob("*.json"):
        m = re.fullmatch(r"sched_\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})\.json", path.name)
        live = (m and _live_ttl(date.fromisoformat(m.group(1))) is not None) or path.name.endswith(f"_{season}.json")
        if live:
            path.unlink(missing_ok=True)
            removed += 1
    return {"season": season, "cleared": removed}


def _cache_write(name: str, data: Any) -> None:
    ensure_dirs()
    (CACHE_DIR / name).write_text(json.dumps(data))


def fetch_schedule_dates(start: date, end: date, *, use_cache: bool = True) -> list[dict]:
    """Fetch schedule games across a date range (inclusive)."""
    key = f"sched_{start.isoformat()}_{end.isoformat()}.json"
    if use_cache:
        cached = _cache_read(key, max_age_hours=_live_ttl(end))
        if cached is not None:
            return cached

    games: list[dict] = []
    # API accepts startDate/endDate
    data = _get(
        "/schedule",
        {
            "sportId": 1,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "hydrate": "probablePitcher,team,linescore,weather",
        },
    )
    for day in data.get("dates", []):
        for g in day.get("games", []):
            games.append(_normalize_game(g, day.get("date")))
    _cache_write(key, games)
    return games


def _normalize_game(g: dict, gameday: str | None) -> dict:
    teams = g.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    home_team = home.get("team", {})
    away_team = away.get("team", {})
    venue = g.get("venue", {})
    weather = g.get("weather") or {}
    status = (g.get("status") or {}).get("detailedState")

    home_score = home.get("score")
    away_score = away.get("score")
    hp = home.get("probablePitcher") or {}
    ap = away.get("probablePitcher") or {}

    return {
        "game_pk": g.get("gamePk"),
        "gameday": gameday or (g.get("officialDate")),
        "game_date": g.get("gameDate"),
        "status": status,
        "home_id": home_team.get("id"),
        "home_team": home_team.get("abbreviation") or home_team.get("teamName"),
        "home_name": home_team.get("name"),
        "away_id": away_team.get("id"),
        "away_team": away_team.get("abbreviation") or away_team.get("teamName"),
        "away_name": away_team.get("name"),
        "home_score": home_score,
        "away_score": away_score,
        "venue": venue.get("name"),
        "venue_id": venue.get("id"),
        "home_sp_id": hp.get("id"),
        "home_sp_name": hp.get("fullName"),
        "away_sp_id": ap.get("id"),
        "away_sp_name": ap.get("fullName"),
        "temp": _parse_temp(weather.get("temp")),
        "wind": weather.get("wind"),
        "condition": weather.get("condition"),
        "double_header": g.get("doubleHeader"),
        "game_number": g.get("gameNumber"),
        "season": int(str(g.get("season") or current_mlb_season())),
    }


def _parse_temp(raw) -> float | None:
    if raw is None:
        return None
    try:
        return float(str(raw).split()[0])
    except (TypeError, ValueError):
        return None


def schedule_frame(start: date, end: date) -> pl.DataFrame:
    games = fetch_schedule_dates(start, end)
    if not games:
        return pl.DataFrame()
    return pl.DataFrame(games)


def fetch_team_season_stats(season: int | None = None) -> pl.DataFrame:
    season = season or current_mlb_season()
    key = f"team_stats_{season}.json"
    cached = _cache_read(key, max_age_hours=_season_ttl(season))
    if cached is not None:
        return pl.DataFrame(cached)

    # Hitting + pitching season leaders by team
    hit = _get(
        "/teams/stats",
        {"season": season, "group": "hitting", "stats": "season", "sportIds": 1},
    )
    pit = _get(
        "/teams/stats",
        {"season": season, "group": "pitching", "stats": "season", "sportIds": 1},
    )
    rows: dict[int, dict] = {}
    for block, prefix in [(hit, "hit"), (pit, "pit")]:
        for entry in block.get("stats", []):
            for split in entry.get("splits", []):
                team = split.get("team", {})
                tid = team.get("id")
                if tid is None:
                    continue
                row = rows.setdefault(
                    tid,
                    {
                        "team_id": tid,
                        "team": team.get("abbreviation") or team.get("name"),
                        "season": season,
                    },
                )
                st = split.get("stat", {})
                if prefix == "hit":
                    row.update(
                        {
                            "runs": float(st.get("runs") or 0),
                            "avg": float(st.get("avg") or 0),
                            "obp": float(st.get("obp") or 0),
                            "slg": float(st.get("slg") or 0),
                            "ops": float(st.get("ops") or 0),
                            "home_runs": float(st.get("homeRuns") or 0),
                            "games_played": float(st.get("gamesPlayed") or 1),
                        }
                    )
                else:
                    row.update(
                        {
                            "era": float(st.get("era") or 4.2),
                            "whip": float(st.get("whip") or 1.3),
                            "strikeouts": float(st.get("strikeOuts") or 0),
                            "walks": float(st.get("baseOnBalls") or 0),
                            "saves": float(st.get("saves") or 0),
                            "holds": float(st.get("holds") or 0),
                            "pit_games": float(st.get("gamesPlayed") or 1),
                        }
                    )

    out = list(rows.values())
    for r in out:
        gp = max(r.get("games_played") or 1, 1)
        r["rpg"] = float(r.get("runs") or 0) / gp
        r["k_bb"] = (r.get("strikeouts") or 0) / max(r.get("walks") or 1, 1)
    _cache_write(key, out)
    return pl.DataFrame(out)


def fetch_pitcher_stats(person_id: int, season: int | None = None) -> dict:
    season = season or current_mlb_season()
    key = f"pitcher_{person_id}_{season}.json"
    cached = _cache_read(key, max_age_hours=_season_ttl(season))
    if cached is not None:
        return cached

    data = _get(
        f"/people/{person_id}/stats",
        {"stats": "season", "group": "pitching", "season": season, "sportId": 1},
    )
    out = {
        "person_id": person_id,
        "season": season,
        "era": 4.50,
        "whip": 1.35,
        "k_per_9": 8.0,
        "bb_per_9": 3.2,
        "ip": 0.0,
        "wins": 0,
        "losses": 0,
        "throws": None,
    }
    # handedness
    try:
        person = _get(f"/people/{person_id}")
        people = person.get("people", [])
        if people:
            out["throws"] = people[0].get("pitchHand", {}).get("code")
            out["name"] = people[0].get("fullName")
    except Exception:
        pass

    for block in data.get("stats", []):
        for split in block.get("splits", []):
            st = split.get("stat", {})
            out.update(
                {
                    "era": float(st.get("era") or out["era"]),
                    "whip": float(st.get("whip") or out["whip"]),
                    "k_per_9": float(st.get("strikeoutsPer9Inn") or out["k_per_9"]),
                    "bb_per_9": float(st.get("walksPer9Inn") or out["bb_per_9"]),
                    "ip": float(st.get("inningsPitched") or out["ip"]),
                    "wins": int(st.get("wins") or 0),
                    "losses": int(st.get("losses") or 0),
                }
            )
    _cache_write(key, out)
    return out


def upcoming_and_recent(days_ahead: int = 3, days_back: int = 45) -> pl.DataFrame:
    today = date.today()
    start = today - timedelta(days=days_back)
    end = today + timedelta(days=days_ahead)
    return schedule_frame(start, end)


def completed_games(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & pl.col("status").is_in(["Final", "Game Over", "Completed Early"])
    )
