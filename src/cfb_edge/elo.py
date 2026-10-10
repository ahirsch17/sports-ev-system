"""
CFB team Elo — higher K and stronger HFA than NFL.

Cupcake / FCS games still update ratings, but with reduced K so blowouts
against weak opponents do not inflate FBS ratings as hard.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))


def build_cfb_elo_table(
    schedules: pl.DataFrame,
    *,
    k: float = 32.0,
    hfa_elo: float = 72.0,
    points_per_elo: float = 0.04,
    cupcake_k_scale: float = 0.35,
    season_regress: float = 0.25,
) -> pl.DataFrame:
    """
    Walk-forward Elo: one row per game with pre-game ratings and predicted margin.
    Updates only after completed games, in chronological order.
    """
    ratings: dict[str, float] = {}
    last_season: int | None = None
    rows: list[dict[str, Any]] = []

    def get(team: str) -> float:
        return ratings.setdefault(team, 1500.0)

    games = schedules.sort(["season", "week", "gameday", "game_id"])
    for r in games.iter_rows(named=True):
        season = int(r["season"])
        if last_season is not None and season != last_season and season_regress > 0:
            mean = 1500.0
            for t, val in list(ratings.items()):
                ratings[t] = val + season_regress * (mean - val)
        last_season = season

        home = str(r["home_team"])
        away = str(r["away_team"])
        rh, ra = get(home), get(away)
        neutral = bool(r.get("neutral_site"))
        hfa = 0.0 if neutral else hfa_elo
        exp_home = expected_score(rh + hfa, ra)
        pred_margin = (rh + hfa - ra) * points_per_elo

        rows.append(
            {
                "game_id": r["game_id"],
                "season": season,
                "week": r.get("week"),
                "gameday": r.get("gameday"),
                "home_team": home,
                "away_team": away,
                "elo_home": rh,
                "elo_away": ra,
                "elo_pred_margin": pred_margin,
                "elo_home_win_prob": exp_home,
                "is_cupcake": bool(r.get("is_cupcake")),
                "both_fbs": bool(r.get("both_fbs", True)),
                "neutral_site": neutral,
            }
        )

        hs, aws = r.get("home_score"), r.get("away_score")
        if hs is None or aws is None:
            continue

        hs_f, aws_f = float(hs), float(aws)
        if hs_f == aws_f:
            actual = 0.5
        else:
            actual = 1.0 if hs_f > aws_f else 0.0
        mov = abs(hs_f - aws_f)
        mult = min(2.0, (mov + 1) ** 0.40)
        k_eff = k * (cupcake_k_scale if r.get("is_cupcake") else 1.0)
        delta = k_eff * mult * (actual - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def latest_ratings_from_table(elo_table: pl.DataFrame) -> dict[str, float]:
    """Latest known Elo per team (from last row mentioning each side)."""
    if elo_table.is_empty():
        return {}
    ratings: dict[str, float] = {}
    for r in elo_table.sort(["season", "week", "gameday", "game_id"]).iter_rows(named=True):
        ratings[str(r["home_team"])] = float(r["elo_home"])
        ratings[str(r["away_team"])] = float(r["elo_away"])
    return ratings
