"""Simple MLB team Elo from completed scores."""

from __future__ import annotations

import polars as pl


def build_mlb_elo(completed: pl.DataFrame, *, k: float = 6.0, hfa: float = 20.0) -> dict[str, float]:
    """Return latest Elo by team abbreviation after processing completed chronologically."""
    ratings: dict[str, float] = {}

    def get(t: str) -> float:
        return ratings.setdefault(t, 1500.0)

    games = completed.sort("gameday")
    for r in games.iter_rows(named=True):
        home = str(r["home_team"]).upper()
        away = str(r["away_team"]).upper()
        rh, ra = get(home), get(away)
        exp = 1.0 / (1.0 + 10 ** (-((rh + hfa) - ra) / 400.0))
        hs, aws = float(r["home_score"]), float(r["away_score"])
        if hs == aws:
            continue
        actual = 1.0 if hs > aws else 0.0
        mov = abs(hs - aws)
        mult = min(1.6, (mov + 0.5) ** 0.4)
        delta = k * mult * (actual - exp)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def elo_run_expectation(elo_off: float, elo_def: float, league_rpg: float = 4.5) -> float:
    """Map Elo mismatch to expected runs."""
    # 40 elo ≈ 0.1 runs
    return league_rpg + (elo_off - elo_def) * 0.0025
