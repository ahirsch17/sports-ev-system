"""DraftKings odds via The Odds API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests

from nfl_edge.config import ODDS_API_KEY, ODDS_BOOK, ODDS_DIR, ensure_dirs

SPORT_KEY = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4"


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ODDS_DIR / f"events_{stamp}.json"
    path.write_text(json.dumps(data, indent=2))
    # also write latest pointer
    (ODDS_DIR / "events_latest.json").write_text(json.dumps(data, indent=2))
    return data


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
