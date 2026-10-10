"""Load and cache nflverse regular-season data."""

from __future__ import annotations

import os
import time
from typing import Sequence

import polars as pl

from nfl_edge.config import CACHE_DIR, START_SEASON, current_nfl_season, ensure_dirs

CACHE_TTL_HOURS = float(os.getenv("NFL_EDGE_CACHE_TTL_HOURS", "6"))


def _season_list(seasons: Sequence[int] | None) -> list[int]:
    if seasons is None:
        end = current_nfl_season()
        return list(range(START_SEASON, end + 1))
    return list(seasons)


def _cache_path(name: str, seasons: Sequence[int]) -> "object":
    tag = f"{min(seasons)}_{max(seasons)}"
    return CACHE_DIR / f"{name}_{tag}.parquet"


def _to_pandas_then_polars(df) -> pl.DataFrame:
    """nflreadpy returns Polars; older shims may return pandas."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df.to_pandas() if hasattr(df, "to_pandas") else df)


def _cache_is_fresh(path, seasons: list[int]) -> bool:
    """Past seasons never change; anything covering the live season expires after the TTL."""
    if current_nfl_season() not in seasons:
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _load_cached(name: str, seasons: list[int], loader) -> pl.DataFrame:
    ensure_dirs()
    path = _cache_path(name, seasons)
    if path.exists() and _cache_is_fresh(path, seasons):
        return pl.read_parquet(path)
    try:
        raw = loader(seasons)
    except Exception:
        if path.exists():
            return pl.read_parquet(path)
        raise
    df = _to_pandas_then_polars(raw)
    df.write_parquet(path)
    return df


def load_schedules(seasons: Sequence[int] | None = None, *, regular_only: bool = True) -> pl.DataFrame:
    import nflreadpy as nfl

    seasons = _season_list(seasons)

    def loader(ss: list[int]):
        return nfl.load_schedules(seasons=ss)

    df = _load_cached("schedules", seasons, loader)
    if regular_only and "game_type" in df.columns:
        df = df.filter(pl.col("game_type") == "REG")
    # Drop unfinished games without scores for historical work
    return df


def load_team_stats(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    import nflreadpy as nfl

    seasons = _season_list(seasons)

    def loader(ss: list[int]):
        return nfl.load_team_stats(seasons=ss)

    return _load_cached("team_stats", seasons, loader)


def load_player_stats(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    import nflreadpy as nfl

    seasons = _season_list(seasons)

    def loader(ss: list[int]):
        # weekly player stats
        try:
            return nfl.load_player_stats(seasons=ss, summary_level="week")
        except TypeError:
            return nfl.load_player_stats(seasons=ss)

    return _load_cached("player_stats", seasons, loader)


def load_injuries(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    import nflreadpy as nfl

    seasons = _season_list(seasons)
    # injuries available since 2009
    seasons = [s for s in seasons if s >= 2009]
    if not seasons:
        return pl.DataFrame()

    def loader(ss: list[int]):
        return nfl.load_injuries(seasons=ss)

    return _load_cached("injuries", seasons, loader)


def load_depth_charts(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    import nflreadpy as nfl

    seasons = _season_list(seasons)

    def loader(ss: list[int]):
        return nfl.load_depth_charts(seasons=ss)

    return _load_cached("depth_charts", seasons, loader)


def load_pbp(seasons: Sequence[int] | None = None) -> pl.DataFrame:
    """Play-by-play — large; prefer recent seasons for ratings."""
    import nflreadpy as nfl

    seasons = _season_list(seasons)

    def loader(ss: list[int]):
        return nfl.load_pbp(seasons=ss)

    return _load_cached("pbp", seasons, loader)


def completed_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Regular-season games with final scores."""
    cols = schedules.columns
    need = {"home_score", "away_score"}
    if not need.issubset(cols):
        return schedules.head(0)
    return schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )


def refresh_all(seasons: Sequence[int] | None = None) -> dict[str, int]:
    """Force re-download by deleting cache then loading."""
    ensure_dirs()
    seasons = _season_list(seasons)
    for path in CACHE_DIR.glob("*.parquet"):
        # only wipe overlapping caches
        path.unlink(missing_ok=True)
    out = {
        "schedules": load_schedules(seasons).height,
        "team_stats": load_team_stats(seasons).height,
        "player_stats": load_player_stats(seasons).height,
        "injuries": load_injuries(seasons).height,
        "depth_charts": load_depth_charts(seasons).height,
    }
    return out
