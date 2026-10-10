"""Build market odds rows from nflverse schedule lines (offline fallback)."""

from __future__ import annotations

from typing import Any

import polars as pl

from nfl_edge.edge.calculator import TEAM_ALIASES


ABBR_TO_FULL = {v: k for k, v in TEAM_ALIASES.items() if "Football" not in k and k not in {"Oakland Raiders", "San Diego Chargers", "St. Louis Rams"}}


def schedule_lines_as_odds(schedules: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Convert nflverse spread/total/ML columns into the same flat shape as The Odds API.

    nflverse `spread_line`: positive ⇒ home favored by that many points.
    We emit American home spread = -spread_line.
    """
    rows: list[dict[str, Any]] = []
    need = {"home_team", "away_team", "spread_line", "total_line"}
    if not need.issubset(set(schedules.columns)):
        return rows

    for r in schedules.iter_rows(named=True):
        home = str(r["home_team"])
        away = str(r["away_team"])
        home_full = ABBR_TO_FULL.get(home, home)
        away_full = ABBR_TO_FULL.get(away, away)
        event_id = str(r.get("game_id") or f"{home}_{away}")

        spread_line = r.get("spread_line")
        if spread_line is not None:
            home_spread = -float(spread_line)  # American home line
            away_spread = float(spread_line)
            home_odds = r.get("home_spread_odds") or -110
            away_odds = r.get("away_spread_odds") or -110
            rows.append(
                {
                    "event_id": event_id,
                    "home_team": home_full,
                    "away_team": away_full,
                    "commence_time": r.get("gameday"),
                    "market": "spreads",
                    "selection": home_full,
                    "price": int(home_odds),
                    "point": home_spread,
                }
            )
            rows.append(
                {
                    "event_id": event_id,
                    "home_team": home_full,
                    "away_team": away_full,
                    "commence_time": r.get("gameday"),
                    "market": "spreads",
                    "selection": away_full,
                    "price": int(away_odds),
                    "point": away_spread,
                }
            )

        total_line = r.get("total_line")
        if total_line is not None:
            # nflverse may not always have over/under prices; use -110
            for side in ("Over", "Under"):
                rows.append(
                    {
                        "event_id": event_id,
                        "home_team": home_full,
                        "away_team": away_full,
                        "commence_time": r.get("gameday"),
                        "market": "totals",
                        "selection": side,
                        "price": -110,
                        "point": float(total_line),
                    }
                )

        if r.get("home_moneyline") is not None and r.get("away_moneyline") is not None:
            rows.append(
                {
                    "event_id": event_id,
                    "home_team": home_full,
                    "away_team": away_full,
                    "commence_time": r.get("gameday"),
                    "market": "h2h",
                    "selection": home_full,
                    "price": int(r["home_moneyline"]),
                    "point": None,
                }
            )
            rows.append(
                {
                    "event_id": event_id,
                    "home_team": home_full,
                    "away_team": away_full,
                    "commence_time": r.get("gameday"),
                    "market": "h2h",
                    "selection": away_full,
                    "price": int(r["away_moneyline"]),
                    "point": None,
                }
            )

    return rows
