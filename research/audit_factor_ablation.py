"""
Walk-forward factor ablation for NFL seasons 2022-2025.

Configs:
  a) baseline Elo+points only (no injuries/deep/situational; weather+form on)
  b) +injuries
  c) +deep similar-opponent
  d) +rest/division/pace (situational)
  e) full model

Writes output/audit_factor_ablation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from nfl_edge.config import OUTPUT_DIR, ensure_dirs
from nfl_edge.features.deep_matchups import build_deep_matchup_features, build_team_style_table
from nfl_edge.features.injuries import build_injury_features
from nfl_edge.features.ratings import build_team_game_log
from nfl_edge.features.situational import build_situational_features
from nfl_edge.features.weather import load_weights
from nfl_edge.ingest.nflverse import (
    load_depth_charts,
    load_injuries,
    load_player_stats,
    load_schedules,
    load_team_stats,
)
from nfl_edge.models.game import backtest_margins, predict_slate


def _gated(preds: pl.DataFrame, conf_min: float) -> dict | None:
    df = preds.filter(
        pl.col("actual_home_score").is_not_null() & pl.col("actual_away_score").is_not_null()
    )
    if df.is_empty():
        return None
    df = df.filter(pl.col("pred_margin").abs() >= conf_min)
    if df.is_empty():
        return {"n": 0, "winner_accuracy": None}
    hit = (df["pred_margin"] > 0) == (df["actual_home_score"] > df["actual_away_score"])
    return {"n": df.height, "winner_accuracy": round(float(hit.mean()), 4)}


def _eval_season(preds: pl.DataFrame, season: int, conf_min: float) -> dict:
    season_preds = preds.filter(
        (pl.col("season") == season) & pl.col("actual_home_score").is_not_null()
    )
    bt = backtest_margins(season_preds)
    return {**bt, "season": season, "conf_gate": _gated(season_preds, conf_min)}


CONFIGS = [
    {
        "id": "a_baseline_elo_points",
        "label": "baseline Elo+points",
        "use_injuries": False,
        "use_deep": False,
        "use_situational": False,
    },
    {
        "id": "b_plus_injuries",
        "label": "+injuries",
        "use_injuries": True,
        "use_deep": False,
        "use_situational": False,
    },
    {
        "id": "c_plus_deep",
        "label": "+deep similar-opponent",
        "use_injuries": False,
        "use_deep": True,
        "use_situational": False,
    },
    {
        "id": "d_plus_situational",
        "label": "+rest/division/pace",
        "use_injuries": False,
        "use_deep": False,
        "use_situational": True,
    },
    {
        "id": "e_full",
        "label": "full model",
        "use_injuries": True,
        "use_deep": True,
        "use_situational": True,
    },
]


def run_ablation(seasons: list[int] | None = None) -> dict:
    seasons = seasons or [2022, 2023, 2024, 2025]
    weights = load_weights()
    conf_min = float((weights.get("confidence") or {}).get("min_abs_margin", 5.0))
    results: dict[str, dict] = {}

    # Preload shared history once (widest window)
    min_hist = min(seasons) - 3
    all_seasons = list(range(min_hist, max(seasons) + 1))
    print(f"Loading schedules/stats for {all_seasons[0]}-{all_seasons[-1]}...")
    schedules_all = load_schedules(all_seasons)
    team_stats_all = load_team_stats(all_seasons)
    player_stats_all = load_player_stats(all_seasons)
    injuries_all = load_injuries(seasons)
    depth_all = load_depth_charts(seasons)
    team_games_all = build_team_game_log(schedules_all)
    styles_all = (
        build_team_style_table(team_stats_all) if not team_stats_all.is_empty() else pl.DataFrame()
    )

    for cfg in CONFIGS:
        t0 = time.time()
        print(f"\n=== {cfg['id']}: {cfg['label']} ===")
        season_rows = []
        all_preds: list[pl.DataFrame] = []

        for season in seasons:
            hist = list(range(season - 2, season + 1))
            sched = schedules_all.filter(pl.col("season").is_in(hist))
            stats = team_stats_all.filter(pl.col("season").is_in(hist))
            season_games = sched.filter(pl.col("season") == season)

            inj_feat = None
            if cfg["use_injuries"]:
                inj = injuries_all.filter(pl.col("season") == season) if not injuries_all.is_empty() else injuries_all
                depth = depth_all
                # Build injuries for the season's games only (join later on game_id)
                inj_feat = build_injury_features(season_games, inj, depth, weights)

            deep = None
            if cfg["use_deep"]:
                ps = player_stats_all.filter(pl.col("season").is_in(hist))
                tg = team_games_all.filter(pl.col("season").is_in(hist))
                print(f"  deep features {season} ({season_games.height} games)...")
                deep = build_deep_matchup_features(
                    season_games, stats, ps, depth_all, tg, weights=weights
                )

            sit = None
            if cfg["use_situational"]:
                sit = build_situational_features(season_games, styles=styles_all, weights=weights)

            preds = predict_slate(
                sched,
                stats,
                inj_feat,
                weights,
                deep_features=deep,
                situational_features=sit,
                use_injuries=cfg["use_injuries"],
                use_deep=cfg["use_deep"],
                use_situational=cfg["use_situational"],
            )
            row = _eval_season(preds, season, conf_min)
            season_rows.append(row)
            print(
                f"  {season}: n={row.get('n')} acc={row.get('winner_accuracy')} "
                f"mae={row.get('mae_margin')} gated={row.get('conf_gate')}"
            )
            all_preds.append(
                preds.filter(
                    (pl.col("season") == season) & pl.col("actual_home_score").is_not_null()
                )
            )

        pooled = pl.concat(all_preds, how="diagonal_relaxed") if all_preds else pl.DataFrame()
        overall = backtest_margins(pooled) if not pooled.is_empty() else {"n": 0}
        gated = _gated(pooled, conf_min) if not pooled.is_empty() else None
        results[cfg["id"]] = {
            "label": cfg["label"],
            "flags": {
                "injuries": cfg["use_injuries"],
                "deep": cfg["use_deep"],
                "situational": cfg["use_situational"],
            },
            "seasons": season_rows,
            "overall": overall,
            "confidence_gated": gated,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    # Rank by winner accuracy then MAE
    ranked = sorted(
        (
            {
                "id": k,
                "label": v["label"],
                "winner_accuracy": v["overall"].get("winner_accuracy"),
                "mae_margin": v["overall"].get("mae_margin"),
                "gated": v.get("confidence_gated"),
            }
            for k, v in results.items()
        ),
        key=lambda x: (-(x["winner_accuracy"] or 0), x["mae_margin"] or 99),
    )

    inventory = {
        "similar_opponent_film_proxy": {
            "status": "modeled",
            "where": "features/deep_matchups.py similar_opponent_signal",
            "notes": "Style-distance comps + roster continuity; rematch boost",
        },
        "player_mismatches_continuity": {
            "status": "modeled",
            "where": "features/deep_matchups.py player_mismatch_adjustments / roster_continuity",
        },
        "injuries": {
            "status": "modeled",
            "where": "features/injuries.py",
            "notes": "QB/WR/RB/OL/CB severity; notes flow to UI factor_breakdown",
        },
        "coaching_play_calling": {
            "status": "proxy",
            "where": "features/situational.py pace_pass_tendency_adj",
            "notes": "Pass-rate + EPA vs opponent sack pressure (not named coaches)",
        },
        "rest_short_week": {
            "status": "modeled",
            "where": "features/situational.py rest_short_week_adj",
            "notes": "Rest diff, short week (<=5), Thursday HFA dampen",
        },
        "division_rivalry_revenge_crowd": {
            "status": "partial",
            "modeled": "division_game HFA dampener",
            "missing": "named rivalries, revenge games, crowd noise beyond HFA",
            "where": "features/situational.py division_game_adj",
        },
        "psychological_letdown_backed_corner": {
            "status": "not_modeled",
            "reason": "No clean pre-game measurable without standings/narrative leakage",
        },
        "elo_weather_momentum": {
            "status": "modeled",
            "where": "features/elo.py, weather.py, ratings form_ema",
        },
    }

    out = {
        "seasons": seasons,
        "confidence_min_abs_margin": conf_min,
        "inventory": inventory,
        "ablations": results,
        "ranked": ranked,
        "notes": (
            "Walk-forward: each season uses prior 2 seasons + current season schedule "
            "with rolling ratings (no future leakage). Deep built on season games only."
        ),
    }
    return out


def main():
    ensure_dirs()
    report = run_ablation()
    path = Path(OUTPUT_DIR) / "audit_factor_ablation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nWrote", path)
    print(json.dumps({"ranked": report["ranked"]}, indent=2))


if __name__ == "__main__":
    main()
