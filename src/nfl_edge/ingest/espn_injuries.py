"""The current week's Out and Doubtful list, when the weekly file does not have it yet."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

import polars as pl

from nfl_edge.config import CACHE_DIR

_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
_CACHE = CACHE_DIR / "espn_injuries.json"
_TTL_SECONDS = 30 * 60
_OUT = {"out", "doubtful"}


def outs_from_payload(payload: dict) -> set[str]:
    """Lower-cased names listed Out or Doubtful. Questionable stays available."""
    names: set[str] = set()
    for team in payload.get("injuries") or []:
        for row in team.get("injuries") or []:
            status = str(row.get("status") or "").strip().lower()
            if status not in _OUT:
                continue
            athlete = row.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip().lower()
            if name:
                names.add(name)
    return names


def _cached_payload() -> dict | None:
    if not _CACHE.exists():
        return None
    try:
        body = json.loads(_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    saved = body.get("saved_at")
    if not isinstance(saved, str):
        return None
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(saved)
    except ValueError:
        return None
    if age.total_seconds() > _TTL_SECONDS:
        return None
    payload = body.get("payload")
    return payload if isinstance(payload, dict) else None


def current_out_doubtful() -> set[str]:
    """Out and Doubtful on the live report. Empty if the report cannot be read."""
    payload = _cached_payload()
    if payload is None:
        try:
            with urllib.request.urlopen(_URL, timeout=20) as response:
                payload = json.loads(response.read().decode())
        except (OSError, json.JSONDecodeError, ValueError):
            return set()
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _CACHE.write_text(
                json.dumps(
                    {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "payload": payload,
                    }
                )
            )
        except OSError:
            pass
    return outs_from_payload(payload)


def week_already_published(injuries: pl.DataFrame | None, season: int, week: int) -> bool:
    """True when the weekly file already has a report for this week."""
    need = {"season", "week", "report_status"}
    return bool(
        injuries is not None
        and not injuries.is_empty()
        and need.issubset(injuries.columns)
        and injuries.filter(
            (pl.col("season") == season)
            & (pl.col("week") == week)
            & pl.col("report_status").is_not_null()
        ).height
        > 0
    )


def rows_from_payload(payload: dict, *, season: int, week: int) -> list[dict]:
    """Out and Doubtful rows in the weekly injury shape. Questionable is left off."""
    rows: list[dict] = []
    for team in payload.get("injuries") or []:
        for row in team.get("injuries") or []:
            status = str(row.get("status") or "").strip().lower()
            if status not in _OUT:
                continue
            athlete = row.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            abbr = str((athlete.get("team") or {}).get("abbreviation") or "").strip().upper()
            pos = str((athlete.get("position") or {}).get("abbreviation") or "").strip().upper()
            if not name or not abbr:
                continue
            eid = str(athlete.get("id") or row.get("id") or name)
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": abbr,
                    "gsis_id": f"espn-{eid}",
                    "full_name": name,
                    "position": pos or None,
                    "report_status": "Out" if status == "out" else "Doubtful",
                    "practice_status": None,
                    "report_primary_injury": None,
                    "practice_primary_injury": None,
                    "report_secondary_injury": None,
                }
            )
    return rows


def supplement_unpublished_week(
    injuries: pl.DataFrame | None,
    season: int,
    week: int,
    payload: dict | None = None,
) -> pl.DataFrame:
    """Add the live Out/Doubtful list when this week is not in the weekly file yet.

    A week that already has report rows is returned unchanged, so a published
    week cannot be rewritten by today's report.
    """
    if week_already_published(injuries, season, week):
        return injuries if injuries is not None else pl.DataFrame()
    if payload is None:
        payload = _cached_payload()
        if payload is None:
            try:
                with urllib.request.urlopen(_URL, timeout=20) as response:
                    payload = json.loads(response.read().decode())
            except (OSError, json.JSONDecodeError, ValueError):
                payload = None
    rows = rows_from_payload(payload or {}, season=season, week=week)
    if not rows:
        return injuries if injuries is not None else pl.DataFrame()
    extra = pl.DataFrame(rows)
    if injuries is None or injuries.is_empty():
        return extra
    return pl.concat([injuries, extra], how="diagonal_relaxed")


def bind_live_rows_to_depth(injuries: pl.DataFrame, depth: pl.DataFrame | None) -> pl.DataFrame:
    """Point a live row at the depth-chart player of the same name, so rank still applies.

    A row that does not match a depth-chart name keeps the report name and stays
    a deep reserve. The weekly file's own ids are left alone.
    """
    if (
        injuries is None
        or injuries.is_empty()
        or depth is None
        or depth.is_empty()
        or "gsis_id" not in injuries.columns
    ):
        return injuries
    from nfl_edge.features.injuries import normalize_depth_charts

    dc = normalize_depth_charts(depth)
    if dc.is_empty() or "full_name" not in dc.columns or "team" not in dc.columns:
        return injuries
    lookup: dict[tuple[str, str], tuple[int, str, str]] = {}
    for row in dc.iter_rows(named=True):
        name = str(row.get("full_name") or "").strip().lower()
        team = str(row.get("team") or "")
        if not name or not team:
            continue
        rank = int(row.get("depth_team") or 99)
        prev = lookup.get((team, name))
        if prev is None or rank < prev[0]:
            lookup[(team, name)] = (rank, str(row.get("gsis_id") or ""), str(row.get("position") or ""))
    if not lookup:
        return injuries

    def _bind(gid: str, team: str, name: str, pos: str) -> tuple[str, str]:
        if not str(gid).startswith("espn-"):
            return gid, pos
        found = lookup.get((str(team), str(name).strip().lower()))
        if found is None or not found[1]:
            return gid, pos
        return found[1], found[2] or pos

    bound = [
        _bind(gid, team, name, pos)
        for gid, team, name, pos in zip(
            injuries["gsis_id"].to_list(),
            injuries["team"].to_list(),
            injuries["full_name"].to_list() if "full_name" in injuries.columns else [""] * injuries.height,
            injuries["position"].to_list() if "position" in injuries.columns else [""] * injuries.height,
        )
    ]
    return injuries.with_columns(
        pl.Series("gsis_id", [item[0] for item in bound]),
        pl.Series("position", [item[1] for item in bound]),
    )


def live_outs_for_week(injuries: pl.DataFrame | None, season: int, week: int) -> set[str]:
    """Use the live report only for a week the weekly file has not published."""
    if week_already_published(injuries, season, week):
        return set()
    return current_out_doubtful()
