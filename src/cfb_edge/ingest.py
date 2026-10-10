"""
CFB schedule ingest.

Preferred source: sportsdataverse cfbfastR-data schedules (CFBD-backed, free GitHub CSV).
Optional: CFBD API if CFBD_API_KEY is set.
REG season; callers filter FBS.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import polars as pl
import requests

from cfb_edge.config import CACHE_DIR, current_cfb_season, ensure_dirs

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "cfb-edge/0.1 (local research)"})

# cfbfastR / sportsdataverse schedule CSV (no API key)
CFBFASTR_CSV = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/"
    "schedules/csv/cfb_schedules_{season}.csv"
)
CFBD_BASE = "https://api.collegefootballdata.com"


def _cache_path(name: str) -> Path:
    ensure_dirs()
    return CACHE_DIR / name


LIVE_CACHE_TTL_HOURS = float(os.getenv("CFB_EDGE_CACHE_TTL_HOURS", "6"))


def _cache_read_bytes(name: str, *, max_age_hours: float | None = None) -> bytes | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    if max_age_hours is not None and (time.time() - path.stat().st_mtime) / 3600 >= max_age_hours:
        return None
    return path.read_bytes()


def _season_ttl(season: int) -> float | None:
    """The live season's schedule gains scores weekly; past seasons are final."""
    return LIVE_CACHE_TTL_HOURS if season >= current_cfb_season() else None


def refresh_live() -> dict:
    """Drop the current season's schedule cache so scores re-download."""
    season = current_cfb_season()
    removed = 0
    for name in (f"cfbfastr_sched_{season}.csv", f"cfbd_sched_{season}.json"):
        path = _cache_path(name)
        if path.exists():
            path.unlink()
            removed += 1
    return {"season": season, "cleared": removed}


def _cache_write_bytes(name: str, data: bytes) -> None:
    _cache_path(name).write_bytes(data)


def data_source_used() -> str:
    if os.getenv("CFBD_API_KEY"):
        return "collegefootballdata.com (API key)"
    return "cfbfastR-data schedules (GitHub CSV / CFBD-backed)"


def _normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Map source columns to a stable internal schema."""
    rename = {}
    if "home_points" in df.columns and "home_score" not in df.columns:
        rename["home_points"] = "home_score"
    if "away_points" in df.columns and "away_score" not in df.columns:
        rename["away_points"] = "away_score"
    if rename:
        df = df.rename(rename)

    # gameday from start_date
    if "gameday" not in df.columns and "start_date" in df.columns:
        df = df.with_columns(
            gameday=pl.col("start_date").cast(pl.Utf8).str.slice(0, 10)
        )

    keep = [
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date",
        "gameday",
        "completed",
        "neutral_site",
        "conference_game",
        "venue",
        "home_id",
        "home_team",
        "home_division",
        "home_conference",
        "home_score",
        "away_id",
        "away_team",
        "away_division",
        "away_conference",
        "away_score",
        "notes",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df.select(cols)

    # dtypes
    casts = []
    for c in ("season", "week", "home_id", "away_id", "game_id"):
        if c in out.columns:
            casts.append(pl.col(c).cast(pl.Int64, strict=False))
    for c in ("home_score", "away_score"):
        if c in out.columns:
            casts.append(pl.col(c).cast(pl.Float64, strict=False))
    for c in ("neutral_site", "conference_game", "completed"):
        if c in out.columns:
            casts.append(pl.col(c).cast(pl.Boolean, strict=False))
    if casts:
        out = out.with_columns(casts)

    # FBS flags
    out = out.with_columns(
        home_is_fbs=(pl.col("home_division").str.to_lowercase() == "fbs")
        if "home_division" in out.columns
        else pl.lit(True),
        away_is_fbs=(pl.col("away_division").str.to_lowercase() == "fbs")
        if "away_division" in out.columns
        else pl.lit(True),
    )
    out = out.with_columns(
        both_fbs=pl.col("home_is_fbs") & pl.col("away_is_fbs"),
        is_cupcake=(
            (pl.col("home_is_fbs") & ~pl.col("away_is_fbs"))
            | (pl.col("away_is_fbs") & ~pl.col("home_is_fbs"))
        ),
    )
    return out


def _load_cfbfastr_season(season: int, *, use_cache: bool = True) -> pl.DataFrame:
    cache_name = f"cfbfastr_sched_{season}.csv"
    raw: bytes | None = None
    if use_cache:
        raw = _cache_read_bytes(cache_name, max_age_hours=_season_ttl(season))
    if raw is None:
        url = CFBFASTR_CSV.format(season=season)
        resp = SESSION.get(url, timeout=90)
        resp.raise_for_status()
        raw = resp.content
        _cache_write_bytes(cache_name, raw)

    df = pl.read_csv(
        raw,
        null_values=["NA", "null", ""],
        infer_schema_length=10000,
        ignore_errors=True,
    )
    return _normalize_frame(df)


def _load_cfbd_season(season: int, *, use_cache: bool = True) -> pl.DataFrame:
    key = os.environ["CFBD_API_KEY"].strip()
    cache_name = f"cfbd_sched_{season}.json"
    data: Any | None = None
    if use_cache:
        cached = _cache_read_bytes(cache_name, max_age_hours=_season_ttl(season))
        if cached is not None:
            data = json.loads(cached.decode())
    if data is None:
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        resp = SESSION.get(
            f"{CFBD_BASE}/games",
            params={"year": season, "seasonType": "both"},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        _cache_write_bytes(cache_name, json.dumps(data).encode())

    rows = []
    for g in data:
        rows.append(
            {
                "game_id": g.get("id"),
                "season": g.get("season"),
                "week": g.get("week"),
                "season_type": g.get("seasonType") or g.get("season_type"),
                "start_date": g.get("startDate") or g.get("start_date"),
                "completed": g.get("completed"),
                "neutral_site": g.get("neutralSite") or g.get("neutral_site"),
                "conference_game": g.get("conferenceGame") or g.get("conference_game"),
                "venue": g.get("venue"),
                "home_id": g.get("homeId") or g.get("home_id"),
                "home_team": g.get("homeTeam") or g.get("home_team"),
                "home_division": g.get("homeDivision") or g.get("home_division"),
                "home_conference": g.get("homeConference") or g.get("home_conference"),
                "home_score": g.get("homePoints") if g.get("homePoints") is not None else g.get("home_points"),
                "away_id": g.get("awayId") or g.get("away_id"),
                "away_team": g.get("awayTeam") or g.get("away_team"),
                "away_division": g.get("awayDivision") or g.get("away_division"),
                "away_conference": g.get("awayConference") or g.get("away_conference"),
                "away_score": g.get("awayPoints") if g.get("awayPoints") is not None else g.get("away_points"),
                "notes": g.get("notes"),
            }
        )
    return _normalize_frame(pl.DataFrame(rows))


def load_schedules(
    seasons: list[int],
    *,
    regular_only: bool = True,
    fbs_only: bool = True,
    include_cupcakes: bool = True,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Load REG (and optionally postseason) FBS schedules.

    fbs_only=True keeps games where at least one side is FBS.
    include_cupcakes=True keeps FBS vs FCS/II/III; set False for both-FBS only.
    """
    frames: list[pl.DataFrame] = []
    use_cfbd = bool(os.getenv("CFBD_API_KEY"))
    for season in seasons:
        if use_cfbd:
            try:
                df = _load_cfbd_season(season, use_cache=use_cache)
            except Exception:
                df = _load_cfbfastr_season(season, use_cache=use_cache)
        else:
            df = _load_cfbfastr_season(season, use_cache=use_cache)
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    out = pl.concat(frames, how="diagonal_relaxed")
    if regular_only and "season_type" in out.columns:
        out = out.filter(pl.col("season_type").str.to_lowercase() == "regular")
    if fbs_only:
        if include_cupcakes:
            out = out.filter(pl.col("home_is_fbs") | pl.col("away_is_fbs"))
        else:
            out = out.filter(pl.col("both_fbs"))
    return out.sort(["season", "week", "gameday", "game_id"])


def completed_games(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
