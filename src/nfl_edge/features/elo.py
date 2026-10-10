"""Elo ratings for NFL teams — walk-forward, no leakage."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class EloState:
    ratings: dict[str, float]
    k: float = 20.0
    hfa_elo: float = 48.0  # ~2.0 points if 1 pt ≈ 24 elo
    points_per_elo: float = 0.04  # margin ≈ elo_diff * this


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))


def build_elo_table(schedules: pl.DataFrame, *, k: float = 20.0, hfa_elo: float = 48.0) -> pl.DataFrame:
    """
    Return one row per game with pre-game elo for home/away and predicted margin.
    Updates ratings after each completed game in chronological order.
    """
    games = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    ).sort(["season", "week", "gameday", "game_id"])

    ratings: dict[str, float] = {}
    rows = []

    def get(team: str) -> float:
        return ratings.setdefault(team, 1505.0)

    for r in games.iter_rows(named=True):
        home = str(r["home_team"])
        away = str(r["away_team"])
        # season reset softening: regress 1/3 toward mean each new season start week 1
        # handled by carrying ratings across seasons (more predictive)

        rh, ra = get(home), get(away)
        exp_home = expected_score(rh + hfa_elo, ra)
        pred_margin = (rh + hfa_elo - ra) * 0.04
        rows.append(
            {
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "home_team": home,
                "away_team": away,
                "elo_home": rh,
                "elo_away": ra,
                "elo_pred_margin": pred_margin,
                "elo_home_win_prob": exp_home,
            }
        )

        # update
        hs, as_ = float(r["home_score"]), float(r["away_score"])
        if hs == as_:
            actual = 0.5
        else:
            actual = 1.0 if hs > as_ else 0.0
        # margin multiplier (cap)
        mov = abs(hs - as_)
        mult = min(1.75, (mov + 1) ** 0.35)
        # underdog upset bonus already in exp
        delta = k * mult * (actual - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    # Also attach elo for unplayed games using latest ratings
    upcoming = schedules.filter(pl.col("home_score").is_null())
    for r in upcoming.iter_rows(named=True):
        home = str(r["home_team"])
        away = str(r["away_team"])
        rh, ra = get(home), get(away)
        exp_home = expected_score(rh + hfa_elo, ra)
        rows.append(
            {
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "home_team": home,
                "away_team": away,
                "elo_home": rh,
                "elo_away": ra,
                "elo_pred_margin": (rh + hfa_elo - ra) * 0.04,
                "elo_home_win_prob": exp_home,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()
