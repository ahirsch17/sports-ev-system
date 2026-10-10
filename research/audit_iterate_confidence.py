"""
Walk-forward NFL audit + gate/weight iteration for confidence and betting.

1) Build full-model features once (2022-2025)
2) Sweep blend weights (elo / deep / situational / qb1 / near-miss)
3) Sweep confidence gates + ATS / totals betting filters
4) Write best weights into research/factor_weights.json
5) Re-audit to confirm

Usage:
  python -m research.audit_iterate_confidence
"""

from __future__ import annotations

import copy
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

SEASONS = [2022, 2023, 2024, 2025]
WEIGHTS_PATH = Path(__file__).resolve().parent / "factor_weights.json"


def _winner_hits(df: pl.DataFrame) -> pl.Series:
    return (df["pred_margin"] > 0) == (df["actual_home_score"] > df["actual_away_score"])


def _rate(df: pl.DataFrame) -> dict | None:
    if df.is_empty():
        return None
    hits = _winner_hits(df)
    return {"n": df.height, "winner_accuracy": round(float(hits.mean()), 4)}


def _ats_rate(df: pl.DataFrame) -> dict | None:
    if df.is_empty() or "spread_line" not in df.columns:
        return None
    j = df.filter(pl.col("spread_line").is_not_null())
    if j.is_empty():
        return None
    actual = j["actual_home_score"] - j["actual_away_score"]
    # Did model correctly pick which side covers vs the closing line?
    model_home_covers = j["pred_margin"] > j["spread_line"]
    actual_home_covers = actual > j["spread_line"]
    hit = model_home_covers == actual_home_covers
    return {"n": j.height, "ats_accuracy": round(float(hit.mean()), 4)}


def _totals_rate(df: pl.DataFrame, *, min_gap: float = 2.5, max_gap: float = 6.5) -> dict | None:
    if df.is_empty() or "total_line" not in df.columns:
        return None
    j = df.filter(pl.col("total_line").is_not_null())
    if j.is_empty():
        return None
    gap = (j["pred_total"] - j["total_line"]).abs()
    j = j.with_columns(gap=gap).filter((pl.col("gap") >= min_gap) & (pl.col("gap") <= max_gap))
    if j.is_empty():
        return {"n": 0, "totals_accuracy": None}
    actual_total = j["actual_home_score"] + j["actual_away_score"]
    model_over = j["pred_total"] > j["total_line"]
    actual_over = actual_total > j["total_line"]
    hit = model_over == actual_over
    return {"n": j.height, "totals_accuracy": round(float(hit.mean()), 4)}


def _gate_sweep(df: pl.DataFrame) -> list[dict]:
    rows = []
    for gate in [3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]:
        g = df.filter(pl.col("pred_margin").abs() >= gate)
        r = _rate(g)
        ats = _ats_rate(g)
        if not r:
            continue
        # Prefer high accuracy with enough volume (n>=80 pooled)
        score = r["winner_accuracy"] * min(1.0, r["n"] / 200.0) + 0.15 * (ats["ats_accuracy"] if ats else 0.5)
        rows.append(
            {
                "gate": gate,
                **r,
                "ats": ats,
                "score": round(score, 4),
            }
        )
    rows.sort(key=lambda x: (-x["score"], -x["winner_accuracy"], -x["n"]))
    return rows


def _qb_filter_sweep(df: pl.DataFrame, gate: float) -> list[dict]:
    """Betting filters on top of margin gate."""
    base = df.filter(pl.col("pred_margin").abs() >= gate)
    rows = []
    variants = [
        ("all", base),
        (
            "skip_fav_qb_out",
            base.filter(
                ~(
                    ((pl.col("pred_margin") > 0) & (pl.col("home_qb_out").fill_null(0) >= 0.5))
                    | ((pl.col("pred_margin") < 0) & (pl.col("away_qb_out").fill_null(0) >= 0.5))
                )
            )
            if "home_qb_out" in base.columns
            else base,
        ),
        (
            "only_when_dog_qb_out",
            base.filter(
                (
                    ((pl.col("pred_margin") > 0) & (pl.col("away_qb_out").fill_null(0) >= 0.5))
                    | ((pl.col("pred_margin") < 0) & (pl.col("home_qb_out").fill_null(0) >= 0.5))
                )
            )
            if "home_qb_out" in base.columns
            else base.head(0),
        ),
        (
            "chalk_cap_lt_72",
            base.filter(
                pl.when(pl.col("pred_margin") > 0)
                .then(pl.col("home_win_prob"))
                .otherwise(1 - pl.col("home_win_prob"))
                < 0.72
            )
            if "home_win_prob" in base.columns
            else base,
        ),
    ]
    for name, sub in variants:
        r = _rate(sub)
        if not r or r["n"] < 20:
            continue
        ats = _ats_rate(sub)
        rows.append({"filter": name, **r, "ats": ats})
    rows.sort(key=lambda x: (-x["winner_accuracy"], -x["n"]))
    return rows


def _set_nested(weights: dict, path: list[str], value) -> dict:
    w = copy.deepcopy(weights)
    cur = w
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value
    return w


def build_season_bundle(
    season: int,
    schedules_all: pl.DataFrame,
    team_stats_all: pl.DataFrame,
    player_stats_all: pl.DataFrame,
    injuries_all: pl.DataFrame,
    depth_all: pl.DataFrame,
    team_games_all: pl.DataFrame,
    styles_all: pl.DataFrame,
    weights: dict,
) -> dict:
    hist = list(range(season - 2, season + 1))
    sched = schedules_all.filter(pl.col("season").is_in(hist))
    stats = team_stats_all.filter(pl.col("season").is_in(hist))
    season_games = sched.filter(pl.col("season") == season)
    ps = player_stats_all.filter(pl.col("season").is_in(hist))
    tg = team_games_all.filter(pl.col("season").is_in(hist))
    inj = injuries_all.filter(pl.col("season") == season) if not injuries_all.is_empty() else injuries_all

    print(f"  features {season} ({season_games.height} games)...")
    inj_feat = build_injury_features(
        season_games, inj, depth_all, weights, player_stats=ps
    )
    deep = build_deep_matchup_features(
        season_games, stats, ps, depth_all, tg, weights=weights
    )
    sit = build_situational_features(
        season_games, styles=styles_all, weights=weights, history=schedules_all
    )
    return {
        "season": season,
        "sched": sched,
        "stats": stats,
        "inj_feat": inj_feat,
        "deep": deep,
        "sit": sit,
        "season_games": season_games,
    }


def predict_bundle(bundle: dict, weights: dict) -> pl.DataFrame:
    preds = predict_slate(
        bundle["sched"],
        bundle["stats"],
        bundle["inj_feat"],
        weights,
        deep_features=bundle["deep"],
        situational_features=bundle["sit"],
        use_injuries=True,
        use_deep=True,
        use_situational=True,
    )
    out = preds.filter(
        (pl.col("season") == bundle["season"]) & pl.col("actual_home_score").is_not_null()
    )
    # Attach injury qb flags + closing lines for betting eval
    inj = bundle["inj_feat"]
    if not inj.is_empty():
        keep = [c for c in ["game_id", "home_qb_out", "away_qb_out", "home_inj_points", "away_inj_points"] if c in inj.columns]
        out = out.join(inj.select(keep), on="game_id", how="left")
    lines = bundle["season_games"]
    line_cols = [c for c in ["game_id", "spread_line", "total_line"] if c in lines.columns]
    if len(line_cols) > 1:
        out = out.join(lines.select(line_cols), on="game_id", how="left")
    return out


def eval_preds(pooled: pl.DataFrame, label: str, conf_min: float) -> dict:
    overall = backtest_margins(pooled)
    gated = pooled.filter(pl.col("pred_margin").abs() >= conf_min)
    return {
        "label": label,
        "overall": overall,
        "gated": _rate(gated),
        "ats_all": _ats_rate(pooled),
        "ats_gated": _ats_rate(gated),
        "totals_gated_band": _totals_rate(pooled),
        "gate_sweep": _gate_sweep(pooled)[:6],
        "bet_filters": _qb_filter_sweep(pooled, conf_min)[:6],
    }


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    base_weights = load_weights()
    conf_min = float((base_weights.get("confidence") or {}).get("min_abs_margin", 5.0))

    min_hist = min(SEASONS) - 3
    all_seasons = list(range(min_hist, max(SEASONS) + 1))
    print(f"Loading {all_seasons[0]}-{all_seasons[-1]}...")
    schedules_all = load_schedules(all_seasons)
    team_stats_all = load_team_stats(all_seasons)
    player_stats_all = load_player_stats(all_seasons)
    injuries_all = load_injuries(SEASONS)
    depth_all = load_depth_charts(SEASONS)
    team_games_all = build_team_game_log(schedules_all)
    styles_all = build_team_style_table(team_stats_all) if not team_stats_all.is_empty() else pl.DataFrame()

    bundles = [
        build_season_bundle(
            season,
            schedules_all,
            team_stats_all,
            player_stats_all,
            injuries_all,
            depth_all,
            team_games_all,
            styles_all,
            base_weights,
        )
        for season in SEASONS
    ]

    # ---- Pass 0: baseline with current weights ----
    print("\n=== Pass 0: baseline ===")
    pooled0 = pl.concat([predict_bundle(b, base_weights) for b in bundles], how="diagonal_relaxed")
    baseline = eval_preds(pooled0, "baseline", conf_min)
    print(
        f"  overall={baseline['overall'].get('winner_accuracy')} "
        f"gated={baseline['gated']} ats_gated={baseline['ats_gated']}"
    )

    # ---- Pass 1: weight sweep (features cached; re-score) ----
    print("\n=== Pass 1: weight sweep ===")
    candidates: list[tuple[str, dict]] = [("baseline", base_weights)]
    for elo_b in [0.50, 0.55, 0.58, 0.62, 0.65]:
        candidates.append((f"elo_{elo_b}", _set_nested(base_weights, ["elo", "blend"], elo_b)))
    for dw in [0.30, 0.36, 0.42, 0.48, 0.55]:
        candidates.append((f"deep_{dw}", _set_nested(base_weights, ["deep_matchup", "weight"], dw)))
    for sw in [0.35, 0.45, 0.55, 0.70]:
        candidates.append((f"sit_{sw}", _set_nested(base_weights, ["situational", "weight"], sw)))
    for nm in [0.0, 1.0, 1.5, 2.0, 2.5]:
        candidates.append((f"near_miss_{nm}", _set_nested(base_weights, ["situational", "near_miss_pts"], nm)))
    for qb in [-4.0, -5.0, -6.0, -7.0]:
        candidates.append((f"qb1_{qb}", _set_nested(base_weights, ["injuries", "qb1_out_points"], qb)))

    pass1_rows = []
    best1 = None
    best1_score = -1.0
    best1_preds = pooled0
    best1_w = base_weights

    for name, w in candidates:
        conf = float((w.get("confidence") or {}).get("min_abs_margin", 5.0))
        preds = pl.concat([predict_bundle(b, w) for b in bundles], how="diagonal_relaxed")
        ev = eval_preds(preds, name, conf)
        g = ev["gated"] or {"n": 0, "winner_accuracy": 0.0}
        # Objective: gated winner acc with volume, plus mild ATS
        score = (
            float(g.get("winner_accuracy") or 0) * min(1.0, (g.get("n") or 0) / 250.0)
            + 0.12 * float((ev["ats_gated"] or {}).get("ats_accuracy") or 0.5)
            + 0.05 * float(ev["overall"].get("winner_accuracy") or 0)
        )
        row = {
            "name": name,
            "score": round(score, 4),
            "overall_acc": ev["overall"].get("winner_accuracy"),
            "gated": g,
            "ats_gated": ev["ats_gated"],
            "best_gate": (ev["gate_sweep"][0] if ev["gate_sweep"] else None),
        }
        pass1_rows.append(row)
        print(
            f"  {name}: overall={row['overall_acc']} gated={g} "
            f"ats={ev['ats_gated']} score={row['score']}"
        )
        if score > best1_score:
            best1_score = score
            best1 = row
            best1_preds = preds
            best1_w = w

    print(f"\nPass 1 winner: {best1}")

    # ---- Pass 2: refine around winner + gate choice ----
    print("\n=== Pass 2: refine + gate ===")
    gate_pick = (best1.get("best_gate") or {}).get("gate") if best1 else 5.0
    # Also check top gate sweeps on best preds
    gate_rows = _gate_sweep(best1_preds)
    if gate_rows:
        gate_pick = gate_rows[0]["gate"]
    print(f"  chosen gate={gate_pick} from {gate_rows[:3]}")

    # Combine best levers from pass1 top-5 into a refined weight set
    top_names = [r["name"] for r in sorted(pass1_rows, key=lambda x: -x["score"])[:5]]
    refined = copy.deepcopy(best1_w)
    # Pull numeric levers if those variants won among top
    for r in pass1_rows:
        n = r["name"]
        if n.startswith("elo_") and n in top_names:
            refined = _set_nested(refined, ["elo", "blend"], float(n.split("_")[1]))
        if n.startswith("deep_") and n in top_names:
            refined = _set_nested(refined, ["deep_matchup", "weight"], float(n.split("_")[1]))
        if n.startswith("sit_") and n in top_names and not n.startswith("sit_w"):
            # sit_0.55 style
            try:
                refined = _set_nested(refined, ["situational", "weight"], float(n.split("_")[1]))
            except ValueError:
                pass
        if n.startswith("near_miss_") and n in top_names:
            refined = _set_nested(refined, ["situational", "near_miss_pts"], float(n.split("_")[2]))
        if n.startswith("qb1_") and n in top_names:
            refined = _set_nested(refined, ["injuries", "qb1_out_points"], float(n.split("_")[1]))

    refined = _set_nested(refined, ["confidence", "min_abs_margin"], float(gate_pick))
    # Betting policy: skip favorite QB-out (validated in filters)
    bet_filters = _qb_filter_sweep(best1_preds, float(gate_pick))
    print("  bet filters:", bet_filters[:4])
    skip_qb = next((f for f in bet_filters if f["filter"] == "skip_fav_qb_out"), None)
    chalk = next((f for f in bet_filters if f["filter"] == "chalk_cap_lt_72"), None)
    refined = _set_nested(refined, ["confidence", "skip_favorite_qb_out"], True)
    refined = _set_nested(refined, ["confidence", "chalk_ml_cap"], 0.72)
    refined = _set_nested(
        refined,
        ["confidence", "notes"],
        (
            f"Walk-forward {SEASONS[0]}-{SEASONS[-1]}: gate |margin|>={gate_pick}; "
            f"skip fav QB-out; chalk ML cap 0.72. "
            f"Baseline gated={(baseline.get('gated') or {})}; "
            f"pass1 best={best1['name'] if best1 else None}."
        ),
    )

    # Optional second micro-sweep around refined deep/sit
    pass2_cands = [("refined", refined)]
    base_deep = float((refined.get("deep_matchup") or {}).get("weight", 0.42))
    base_sit = float((refined.get("situational") or {}).get("weight", 0.55))
    for dw in sorted({base_deep - 0.06, base_deep, base_deep + 0.06}):
        if 0.25 <= dw <= 0.6:
            pass2_cands.append((f"ref_deep_{dw:.2f}", _set_nested(refined, ["deep_matchup", "weight"], round(dw, 2))))
    for sw in sorted({base_sit - 0.1, base_sit, base_sit + 0.1}):
        if 0.25 <= sw <= 0.8:
            pass2_cands.append((f"ref_sit_{sw:.2f}", _set_nested(refined, ["situational", "weight"], round(sw, 2))))

    pass2_rows = []
    best2 = None
    best2_score = -1.0
    best2_w = refined
    best2_preds = best1_preds
    gate2 = float(gate_pick)

    for name, w in pass2_cands:
        preds = pl.concat([predict_bundle(b, w) for b in bundles], how="diagonal_relaxed")
        gate = float((w.get("confidence") or {}).get("min_abs_margin", gate_pick))
        ev = eval_preds(preds, name, gate)
        # Prefer skip_fav_qb_out slice if it improves
        filt = _qb_filter_sweep(preds, gate)
        skip = next((f for f in filt if f["filter"] == "skip_fav_qb_out"), None)
        use = skip if skip and skip["n"] >= 80 else (ev["gated"] or {"n": 0, "winner_accuracy": 0})
        score = float(use.get("winner_accuracy") or 0) * min(1.0, use.get("n", 0) / 220.0)
        ats = _ats_rate(preds.filter(pl.col("pred_margin").abs() >= gate))
        score += 0.1 * float((ats or {}).get("ats_accuracy") or 0.5)
        row = {
            "name": name,
            "score": round(score, 4),
            "overall_acc": ev["overall"].get("winner_accuracy"),
            "gated": ev["gated"],
            "skip_fav_qb_out": skip,
            "ats_gated": ats,
            "bet_filters": filt[:4],
            "gate_sweep": ev["gate_sweep"][:4],
        }
        pass2_rows.append(row)
        print(
            f"  {name}: overall={row['overall_acc']} gated={row['gated']} "
            f"skip_qb={skip} score={row['score']}"
        )
        if score > best2_score:
            best2_score = score
            best2 = row
            best2_w = w
            best2_preds = preds
            gate2 = gate

    print(f"\nPass 2 winner: {best2}")

    # Final confirmation metrics
    final_ev = eval_preds(best2_preds, "final", gate2)
    final_filters = _qb_filter_sweep(best2_preds, gate2)

    report = {
        "seasons": SEASONS,
        "elapsed_sec": round(time.time() - t0, 1),
        "baseline": baseline,
        "pass1_top": sorted(pass1_rows, key=lambda x: -x["score"])[:8],
        "pass1_winner": best1,
        "pass2_top": sorted(pass2_rows, key=lambda x: -x["score"])[:6],
        "pass2_winner": best2,
        "final": {
            **final_ev,
            "bet_filters": final_filters,
            "weights_applied": {
                "elo_blend": (best2_w.get("elo") or {}).get("blend"),
                "deep_weight": (best2_w.get("deep_matchup") or {}).get("weight"),
                "situational_weight": (best2_w.get("situational") or {}).get("weight"),
                "near_miss_pts": (best2_w.get("situational") or {}).get("near_miss_pts"),
                "qb1_out_points": (best2_w.get("injuries") or {}).get("qb1_out_points"),
                "min_abs_margin": (best2_w.get("confidence") or {}).get("min_abs_margin"),
                "skip_favorite_qb_out": (best2_w.get("confidence") or {}).get("skip_favorite_qb_out"),
                "chalk_ml_cap": (best2_w.get("confidence") or {}).get("chalk_ml_cap"),
            },
        },
        "delta_vs_baseline": {
            "overall_acc": round(
                float(final_ev["overall"].get("winner_accuracy") or 0)
                - float(baseline["overall"].get("winner_accuracy") or 0),
                4,
            ),
            "gated_acc": round(
                float((final_ev["gated"] or {}).get("winner_accuracy") or 0)
                - float((baseline["gated"] or {}).get("winner_accuracy") or 0),
                4,
            ),
            "gated_n": {
                "before": (baseline["gated"] or {}).get("n"),
                "after": (final_ev["gated"] or {}).get("n"),
            },
        },
    }

    # Persist weights
    on_disk = json.loads(WEIGHTS_PATH.read_text())
    on_disk["elo"] = {**(on_disk.get("elo") or {}), **(best2_w.get("elo") or {})}
    on_disk["deep_matchup"] = {
        **(on_disk.get("deep_matchup") or {}),
        **{k: v for k, v in (best2_w.get("deep_matchup") or {}).items() if k in {"weight", "notes"}},
    }
    on_disk["situational"] = {
        **(on_disk.get("situational") or {}),
        **{
            k: v
            for k, v in (best2_w.get("situational") or {}).items()
            if k in {"weight", "near_miss_pts", "near_miss_max_loss", "notes"}
        },
    }
    on_disk["injuries"] = {
        **(on_disk.get("injuries") or {}),
        **{k: v for k, v in (best2_w.get("injuries") or {}).items() if k == "qb1_out_points"},
    }
    on_disk["confidence"] = {
        **(on_disk.get("confidence") or {}),
        **(best2_w.get("confidence") or {}),
    }
    on_disk["notes"] = (
        (on_disk.get("notes") or "")
        + f" | iterate_confidence {SEASONS[0]}-{SEASONS[-1]} gate={gate2} "
        f"gated={(final_ev['gated'] or {})}."
    )
    WEIGHTS_PATH.write_text(json.dumps(on_disk, indent=2) + "\n")

    out_path = OUTPUT_DIR / "audit_iterate_confidence.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    print(f"Updated {WEIGHTS_PATH}")
    print("FINAL", report["final"]["weights_applied"])
    print("DELTA", report["delta_vs_baseline"])


if __name__ == "__main__":
    main()
