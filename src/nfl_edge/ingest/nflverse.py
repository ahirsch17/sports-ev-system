"""Load and cache nflverse regular-season data.

Each dataset is cached one season per file. Completed seasons are downloaded once;
the current season is re-downloaded when its file is older than LIVE_TTL_SECONDS
(or immediately via `refresh_current`) so new scores, injuries and depth charts
flow into the ratings.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Callable, Sequence

import polars as pl

from nfl_edge.config import CACHE_DIR, LIVE_TTL_SECONDS, START_SEASON, current_nfl_season, ensure_dirs

log = logging.getLogger(__name__)

DATASETS = ("schedules", "team_stats", "player_stats", "injuries", "depth_charts")
_LEGACY_RANGE_FILE = re.compile(rf"^({'|'.join(DATASETS)}|pbp)_\d{{4}}_\d{{4}}\.parquet$")


def _season_list(seasons: Sequence[int] | None) -> list[int]:
    if seasons is None:
        end = current_nfl_season()
        return list(range(START_SEASON, end + 1))
    return sorted(set(int(s) for s in seasons))


def _cache_path(name: str, season: int):
    return CACHE_DIR / f"{name}_{season}.parquet"


def _to_polars(df) -> pl.DataFrame:
    """nflreadpy returns Polars; older shims may return pandas."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df.to_pandas() if hasattr(df, "to_pandas") else df)


def _is_fresh(path, season: int) -> bool:
    if not path.exists():
        return False
    if season < current_nfl_season():
        return True
    return time.time() - path.stat().st_mtime < LIVE_TTL_SECONDS


def _load_season(name: str, season: int, loader: Callable, *, force: bool) -> pl.DataFrame:
    path = _cache_path(name, season)
    if not force and _is_fresh(path, season):
        return pl.read_parquet(path)
    if season >= current_nfl_season():
        import nflreadpy as nfl

        # nflreadpy keeps its own 24h in-memory copy; drop it so a refresh really re-downloads.
        nfl.clear_cache()
    try:
        df = _to_polars(loader([season]))
    except Exception as exc:
        if path.exists():
            log.warning("Could not refresh %s %s (%s); using cached copy", name, season, exc)
            return pl.read_parquet(path)
        raise
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return df


def _load_cached(name: str, seasons: list[int], loader: Callable, *, force: bool = False) -> pl.DataFrame:
    ensure_dirs()
    frames = [_load_season(name, s, loader, force=force) for s in seasons]
    frames = [f for f in frames if f.width]
    if not frames:
        return pl.DataFrame()
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")


def load_schedules(
    seasons: Sequence[int] | None = None, *, regular_only: bool = True, force: bool = False
) -> pl.DataFrame:
    import nflreadpy as nfl

    df = _load_cached("schedules", _season_list(seasons), lambda ss: nfl.load_schedules(seasons=ss), force=force)
    if regular_only and "game_type" in df.columns:
        df = df.filter(pl.col("game_type") == "REG")
    return df


def load_team_stats(seasons: Sequence[int] | None = None, *, force: bool = False) -> pl.DataFrame:
    import nflreadpy as nfl

    return _load_cached("team_stats", _season_list(seasons), lambda ss: nfl.load_team_stats(seasons=ss), force=force)


def load_player_stats(seasons: Sequence[int] | None = None, *, force: bool = False) -> pl.DataFrame:
    import nflreadpy as nfl

    def loader(ss: list[int]):
        try:
            return nfl.load_player_stats(seasons=ss, summary_level="week")
        except TypeError:
            return nfl.load_player_stats(seasons=ss)

    return _load_cached("player_stats", _season_list(seasons), loader, force=force)


def load_injuries(seasons: Sequence[int] | None = None, *, force: bool = False) -> pl.DataFrame:
    import nflreadpy as nfl

    # injuries available since 2009
    seasons = [s for s in _season_list(seasons) if s >= 2009]
    if not seasons:
        return pl.DataFrame()
    return _load_cached("injuries", seasons, lambda ss: nfl.load_injuries(seasons=ss), force=force)


def load_depth_charts(seasons: Sequence[int] | None = None, *, force: bool = False) -> pl.DataFrame:
    import nflreadpy as nfl

    return _load_cached("depth_charts", _season_list(seasons), lambda ss: nfl.load_depth_charts(seasons=ss), force=force)


def load_pbp(seasons: Sequence[int] | None = None, *, force: bool = False) -> pl.DataFrame:
    """Play-by-play — large; prefer recent seasons for ratings."""
    import nflreadpy as nfl

    return _load_cached("pbp", _season_list(seasons), lambda ss: nfl.load_pbp(seasons=ss), force=force)


def completed_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Regular-season games with final scores."""
    cols = schedules.columns
    need = {"home_score", "away_score"}
    if not need.issubset(cols):
        return schedules.head(0)
    return schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )


_LOADERS: dict[str, Callable[..., pl.DataFrame]] = {
    "schedules": load_schedules,
    "team_stats": load_team_stats,
    "player_stats": load_player_stats,
    "injuries": load_injuries,
    "depth_charts": load_depth_charts,
}


def refresh_all(seasons: Sequence[int] | None = None) -> dict[str, int]:
    """Re-download the given seasons (default: all) without touching other caches."""
    seasons = _season_list(seasons)
    return {name: loader(seasons, force=True).height for name, loader in _LOADERS.items()}


def refresh_current(season: int | None = None) -> dict:
    """Re-download every current-season dataset now and report what changed."""
    ensure_dirs()
    season = season or current_nfl_season()
    for path in CACHE_DIR.glob("*.parquet"):
        if _LEGACY_RANGE_FILE.match(path.name):
            path.unlink(missing_ok=True)

    before = _final_game_count(season)
    rows: dict[str, int] = {}
    errors: dict[str, str] = {}
    for name, loader in _LOADERS.items():
        try:
            rows[name] = loader([season], force=True).height
        except Exception as exc:  # one dataset failing should not block the rest
            errors[name] = str(exc)[:200]
    after = _final_game_count(season)
    return {
        "season": season,
        "rows": rows,
        "errors": errors,
        "final_games": after,
        "new_final_games": max(0, after - before),
        "as_of": data_as_of(season),
    }


def _final_game_count(season: int) -> int:
    path = _cache_path("schedules", season)
    if not path.exists():
        return 0
    df = pl.read_parquet(path)
    if "home_score" not in df.columns:
        return 0
    return completed_games(df.filter(pl.col("game_type") == "REG") if "game_type" in df.columns else df).height


def data_as_of(season: int | None = None) -> str | None:
    """UTC time the current-season schedule (scores) was last downloaded."""
    path = _cache_path("schedules", season or current_nfl_season())
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
