from __future__ import annotations

import os
import time

import polars as pl
import pytest

from nfl_edge.ingest import nflverse

TTL_HOURS = nflverse.LIVE_TTL_SECONDS / 3600


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(nflverse, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(nflverse, "ensure_dirs", lambda: None)
    monkeypatch.setattr(nflverse, "current_nfl_season", lambda: 2026)
    return tmp_path


def _age(path, hours: float) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_live_season_cache_expires(cache_dir):
    calls = []

    def loader(seasons):
        calls.append(seasons)
        return pl.DataFrame({"season": seasons, "n": [len(calls)]})

    nflverse._load_cached("schedules", [2025, 2026], loader)
    assert calls == [[2025], [2026]]
    _age(nflverse._cache_path("schedules", 2025), TTL_HOURS + 1)
    _age(nflverse._cache_path("schedules", 2026), TTL_HOURS + 1)
    df = nflverse._load_cached("schedules", [2025, 2026], loader)
    assert calls[-1] == [2026] and len(calls) == 3
    assert df.filter(pl.col("season") == 2026)["n"][0] == 3


def test_past_seasons_cache_never_expires(cache_dir):
    calls = []

    def loader(seasons):
        calls.append(seasons)
        return pl.DataFrame({"n": [1]})

    nflverse._load_cached("schedules", [2022, 2023], loader)
    _age(nflverse._cache_path("schedules", 2022), 24 * 365)
    nflverse._load_cached("schedules", [2022, 2023], loader)
    assert len(calls) == 2


def test_force_redownloads_fresh_live_season(cache_dir):
    calls = []

    def loader(seasons):
        calls.append(seasons)
        return pl.DataFrame({"n": [len(calls)]})

    nflverse._load_cached("schedules", [2026], loader)
    df = nflverse._load_cached("schedules", [2026], loader, force=True)
    assert len(calls) == 2
    assert df["n"][0] == 2


def test_stale_cache_used_when_download_fails(cache_dir):
    nflverse._load_cached("schedules", [2026], lambda s: pl.DataFrame({"n": [7]}))
    _age(nflverse._cache_path("schedules", 2026), TTL_HOURS + 1)

    def broken(_seasons):
        raise ConnectionError("offline")

    assert nflverse._load_cached("schedules", [2026], broken)["n"][0] == 7
