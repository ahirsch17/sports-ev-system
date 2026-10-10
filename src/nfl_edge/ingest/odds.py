"""DraftKings odds via The Odds API."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import requests

from nfl_edge.config import (
    ODDS_API_KEY,
    ODDS_BOOK,
    ODDS_DIR,
    ODDS_MIN_REFETCH_SECONDS,
    ODDS_TTL_SECONDS,
    ensure_dirs,
)

SPORT_KEY = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4"
_META_PATH = ODDS_DIR / "events_latest.meta.json"


class OddsAPIError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    return {"Accept": "application/json"}


def _require_key() -> str:
    if not ODDS_API_KEY:
        raise OddsAPIError(
            "ODDS_API_KEY not set. Copy .env.example to .env and add a key from the-odds-api.com"
        )
    return ODDS_API_KEY


def fetch_event_odds(
    *,
    markets: str = "h2h,spreads,totals",
    regions: str = "us",
    odds_format: str = "american",
    book: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch main markets for upcoming NFL games from DraftKings (default)."""
    key = _require_key()
    book = book or ODDS_BOOK
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "bookmakers": book,
    }
    url = f"{BASE}/sports/{SPORT_KEY}/odds"
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    if resp.status_code != 200:
        raise OddsAPIError(f"Odds API {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    ensure_dirs()
    now = datetime.now(timezone.utc)
    path = ODDS_DIR / f"events_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(data, indent=2))
    # also write latest pointer
    (ODDS_DIR / "events_latest.json").write_text(json.dumps(data, indent=2))
    _META_PATH.write_text(
        json.dumps(
            {
                "fetched_at": now.isoformat(timespec="seconds"),
                "book": book,
                "requests_remaining": _int_header(resp, "x-requests-remaining"),
                "requests_used": _int_header(resp, "x-requests-used"),
            }
        )
    )
    return data


def _int_header(resp: requests.Response, name: str) -> int | None:
    try:
        return int(float(resp.headers.get(name, "")))
    except ValueError:
        return None


def _read_meta() -> dict[str, Any]:
    if not _META_PATH.exists():
        return {}
    try:
        return json.loads(_META_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _not_started(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop in-progress games: live in-game prices aren't comparable to a pregame model."""
    now = datetime.now(timezone.utc)
    out = []
    for ev in events:
        try:
            start = datetime.fromisoformat(str(ev.get("commence_time")).replace("Z", "+00:00"))
        except ValueError:
            out.append(ev)
            continue
        if start > now:
            out.append(ev)
    return out


def load_live_events(*, force: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    DraftKings main lines for games that haven't kicked off, plus fetch metadata.

    Reuses the saved pull while it is younger than ODDS_TTL_SECONDS (or
    ODDS_MIN_REFETCH_SECONDS when forced) to stay inside the free API quota.
    Returns ([], meta) when no key is configured or the API is unreachable.
    """
    if not ODDS_API_KEY:
        return [], {"error": "ODDS_API_KEY not set"}
    latest = ODDS_DIR / "events_latest.json"
    age = time.time() - latest.stat().st_mtime if latest.exists() else None
    max_age = ODDS_MIN_REFETCH_SECONDS if force else ODDS_TTL_SECONDS
    if age is not None and age < max_age:
        return _not_started(load_cached_events()), {**_read_meta(), "cached": True}
    try:
        return _not_started(fetch_event_odds()), _read_meta()
    except (OddsAPIError, requests.RequestException) as exc:
        if age is not None and age < 24 * 3600:
            return _not_started(load_cached_events()), {**_read_meta(), "cached": True, "error": str(exc)[:200]}
        return [], {"error": str(exc)[:200]}


def fetch_player_props(event_id: str, *, book: str | None = None) -> dict[str, Any]:
    """Fetch player prop markets for one event (uses more API quota)."""
    key = _require_key()
    book = book or ODDS_BOOK
    markets = ",".join(
        [
            "player_pass_yds",
            "player_rush_yds",
            "player_reception_yds",
            "player_pass_tds",
            "player_anytime_td",
        ]
    )
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": book,
    }
    url = f"{BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    if resp.status_code != 200:
        raise OddsAPIError(f"Odds API props {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    ensure_dirs()
    path = ODDS_DIR / f"props_{event_id}.json"
    path.write_text(json.dumps(data, indent=2))
    return data


def load_cached_events(path: str | None = None) -> list[dict[str, Any]]:
    ensure_dirs()
    p = ODDS_DIR / (path or "events_latest.json")
    if not p.exists():
        return []
    return json.loads(p.read_text())


def flatten_main_markets(events: list[dict[str, Any]], book: str | None = None) -> list[dict[str, Any]]:
    """Normalize DK main lines into flat rows for edge comparison."""
    book = book or ODDS_BOOK
    rows: list[dict[str, Any]] = []
    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        event_id = ev.get("id")
        commence = ev.get("commence_time")
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != book:
                continue
            for market in bm.get("markets", []):
                mkey = market.get("key")
                outcomes = market.get("outcomes", [])
                if mkey == "h2h":
                    for o in outcomes:
                        rows.append(
                            {
                                "event_id": event_id,
                                "home_team": home,
                                "away_team": away,
                                "commence_time": commence,
                                "market": "h2h",
                                "selection": o.get("name"),
                                "price": o.get("price"),
                                "point": None,
                            }
                        )
                elif mkey == "spreads":
                    for o in outcomes:
                        rows.append(
                            {
                                "event_id": event_id,
                                "home_team": home,
                                "away_team": away,
                                "commence_time": commence,
                                "market": "spreads",
                                "selection": o.get("name"),
                                "price": o.get("price"),
                                "point": o.get("point"),
                            }
                        )
                elif mkey == "totals":
                    for o in outcomes:
                        rows.append(
                            {
                                "event_id": event_id,
                                "home_team": home,
                                "away_team": away,
                                "commence_time": commence,
                                "market": "totals",
                                "selection": o.get("name"),  # Over / Under
                                "price": o.get("price"),
                                "point": o.get("point"),
                            }
                        )
    return rows


def flatten_prop_markets(prop_payload: dict[str, Any], book: str | None = None) -> list[dict[str, Any]]:
    book = book or ODDS_BOOK
    rows: list[dict[str, Any]] = []
    home = prop_payload.get("home_team")
    away = prop_payload.get("away_team")
    event_id = prop_payload.get("id")
    for bm in prop_payload.get("bookmakers", []):
        if bm.get("key") != book:
            continue
        for market in bm.get("markets", []):
            mkey = market.get("key")
            for o in market.get("outcomes", []):
                rows.append(
                    {
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "market": mkey,
                        "selection": o.get("name"),  # Over / Under / Yes
                        "player": o.get("description"),
                        "price": o.get("price"),
                        "point": o.get("point"),
                    }
                )
    return rows
