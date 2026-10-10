"""Fit how much weight the nfl_edge margin deserves on top of the closing market.

For each completed game: mu(w) = w * model_margin + (1 - w) * market_margin, where
market_margin is nflverse's closing spread_line (expected home margin). Reports, per w:
  - margin MSE vs actual
  - ATS log loss / accuracy vs the closing line (w=0 is a coin flip by construction)
Writes the best ATS log-loss weight to research/factor_weights.json["market_blend"].

Usage: python research/fit_market_blend.py [--seasons 2019 2020 ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_edge.config import WEIGHTS_PATH  # noqa: E402
from nfl_edge.features.weather import load_weights  # noqa: E402
from nfl_edge.ingest.nflverse import load_schedules, load_team_stats  # noqa: E402
from nfl_edge.models.game import predict_slate  # noqa: E402
from sports_ev.models.nfl_edge_strategy import side_cover_probs  # noqa: E402


def collect(seasons: list[int], weights: dict) -> pl.DataFrame:
    frames = []
    for season in seasons:
        hist = list(range(season - 2, season + 1))
        sched = load_schedules(hist)
        stats = load_team_stats(hist)
        preds = predict_slate(sched, stats, None, weights)
        season_preds = preds.filter(
            (pl.col("season") == season) & pl.col("actual_home_score").is_not_null()
        ).join(sched.select(["game_id", "spread_line"]), on="game_id", how="left")
        frames.append(
            season_preds.filter(pl.col("spread_line").is_not_null()).select(
                [
                    "season",
                    "pred_margin",
                    "spread_line",
                    (pl.col("actual_home_score") - pl.col("actual_away_score")).alias("actual_margin"),
                ]
            )
        )
    return pl.concat(frames)


def evaluate(df: pl.DataFrame, w: float, sigma: float) -> dict:
    mse = 0.0
    ll = 0.0
    hits = 0
    n_ats = 0
    for pred, line_margin, actual in df.select(["pred_margin", "spread_line", "actual_margin"]).iter_rows():
        mu = w * pred + (1 - w) * line_margin
        mse += (actual - mu) ** 2
        if actual == line_margin:
            continue
        p_home, _ = side_cover_probs(mu, -line_margin, sigma)
        covered = actual > line_margin
        p = min(max(p_home if covered else 1 - p_home, 1e-9), 1.0)
        ll -= math.log(p)
        hits += int((p_home > 0.5) == covered) if p_home != 0.5 else 0
        n_ats += 1
    return {
        "w": round(w, 2),
        "margin_rmse": round(math.sqrt(mse / df.height), 3),
        "ats_logloss": round(ll / max(n_ats, 1), 5),
        "ats_acc": round(hits / max(n_ats, 1), 4),
        "n": df.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    weights = load_weights()
    sigma = float(weights.get("margin_sigma", 13.5))
    df = collect(args.seasons, weights)
    grid = [i / 20 for i in range(0, 21)]
    results = [evaluate(df, w, sigma) for w in grid]
    for r in results:
        print(r)
    best = min(results, key=lambda r: r["ats_logloss"])
    print("best by ATS log loss:", best)

    by_season = {}
    for season in args.seasons:
        sub = df.filter(pl.col("season") == season)
        if sub.height:
            by_season[season] = evaluate(sub, best["w"], sigma)
    print(json.dumps(by_season, indent=2))

    if not args.no_write:
        data = json.loads(WEIGHTS_PATH.read_text()) if WEIGHTS_PATH.exists() else {}
        data["market_blend"] = {
            "model_weight": best["w"],
            "seasons": args.seasons,
            "n": best["n"],
            "ats_logloss": best["ats_logloss"],
            "ats_logloss_market_only": results[0]["ats_logloss"],
            "notes": "Blend of nfl_edge margin with closing-market margin used by sports_ev NflEdgeSpreadStrategy.",
        }
        WEIGHTS_PATH.write_text(json.dumps(data, indent=2) + "\n")
        print("Updated", WEIGHTS_PATH)


if __name__ == "__main__":
    main()
