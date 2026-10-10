"""
Walk-forward CFB audit + ablation over prior seasons.

Writes output/cfb_audit_report.json and updates research/cfb/factor_weights.json
with the chosen K / HFA / blend / confidence gate.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfb_edge.config import OUTPUT_DIR, RESEARCH_DIR, ensure_dirs
from cfb_edge.ingest import data_source_used, load_schedules
from cfb_edge.predict import backtest_margins, gated_accuracy, predict_games_walkforward
from cfb_edge.weights import DEFAULT_WEIGHTS, load_weights, save_weights


def _rate(hits: list[int]) -> dict | None:
    if not hits:
        return None
    return {"n": len(hits), "winner_accuracy": round(sum(hits) / len(hits), 4)}


def _eval_preds(preds: pl.DataFrame, gate: float) -> dict:
    bt = backtest_margins(preds, both_fbs_only=True)
    gates = gated_accuracy(preds, thresholds=[7.0, 10.0, 14.0, gate], both_fbs_only=True)
    # primary gate key
    key = f"gate_{int(gate)}" if float(gate).is_integer() else f"gate_{gate}"
    # gated_accuracy uses int keys for 7/10/14
    g = gates.get(f"gate_{int(gate)}") or gates.get("gate_10")
    return {
        "overall": bt,
        "gates": gates,
        "primary_gate": g,
        "gated_accuracy": (g or {}).get("winner_accuracy"),
        "gated_n": (g or {}).get("n"),
    }


def run_ablation(schedules: pl.DataFrame, seasons: list[int]) -> dict:
    """Grid search K, HFA, blend, gate; return best config + ablation table."""
    base = deepcopy(DEFAULT_WEIGHTS)
    results = []

    # Ablation families (compact grid; refine around winners)
    configs = []
    # Elo-only
    for k in (24, 32, 40):
        for hfa in (55, 72, 90):
            configs.append(
                {
                    "name": f"elo_k{k}_hfa{hfa}",
                    "family": "elo_only",
                    "elo": {
                        "k": float(k),
                        "hfa_elo": float(hfa),
                        "blend": 1.0,
                        "points_per_elo": 0.04,
                        "cupcake_k_scale": 0.35,
                        "season_regress": 0.25,
                    },
                    "use_form": False,
                    "use_rest": False,
                    "use_sim": False,
                }
            )
    # Elo + form
    for k in (28, 34, 40):
        for hfa in (65, 78, 90):
            for blend in (0.60, 0.75, 0.90):
                configs.append(
                    {
                        "name": f"elo_form_k{k}_hfa{hfa}_b{blend}",
                        "family": "elo_form",
                        "elo": {
                            "k": float(k),
                            "hfa_elo": float(hfa),
                            "blend": float(blend),
                            "points_per_elo": 0.04,
                            "cupcake_k_scale": 0.35,
                            "season_regress": 0.25,
                        },
                        "use_form": True,
                        "use_rest": False,
                        "use_sim": False,
                    }
                )
    # Full (+rest/conf/sim)
    for k in (32, 38):
        for hfa in (72, 85):
            for blend in (0.65, 0.80):
                configs.append(
                    {
                        "name": f"full_k{k}_hfa{hfa}_b{blend}",
                        "family": "full",
                        "elo": {
                            "k": float(k),
                            "hfa_elo": float(hfa),
                            "blend": float(blend),
                            "points_per_elo": 0.04,
                            "cupcake_k_scale": 0.35,
                            "season_regress": 0.25,
                        },
                        "use_form": True,
                        "use_rest": True,
                        "use_sim": True,
                    }
                )

    # Deduplicate by name
    seen = set()
    uniq = []
    for c in configs:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        uniq.append(c)

    print(f"Ablating {len(uniq)} configs over seasons {seasons}…")
    best = None
    for i, cfg in enumerate(uniq):
        w = deepcopy(base)
        w["elo"].update(cfg["elo"])
        if not cfg["use_form"]:
            w["elo"]["blend"] = 1.0
            w["factors"]["recent_form"]["weight"] = 0.0
        if not cfg["use_rest"]:
            w["factors"]["rest"]["weight"] = 0.0
            w["factors"]["conference_rivalry"]["weight"] = 0.0
        if not cfg["use_sim"]:
            w["factors"]["similar_opponent"]["weight"] = 0.0
            w["similar_scale"] = 0.0

        preds = predict_games_walkforward(schedules, w)
        season_rows = []
        all_hits: list[int] = []
        gate_hits = {7: [], 10: [], 14: []}
        for season in seasons:
            sp = preds.filter(pl.col("season") == season)
            ev = _eval_preds(sp, gate=10.0)
            season_rows.append({"season": season, **ev["overall"], "gated": ev["gates"]})
            # collect overall hits
            df = sp.filter(
                pl.col("actual_home_score").is_not_null()
                & pl.col("actual_away_score").is_not_null()
                & (pl.col("both_fbs") == True)  # noqa: E712
            )
            if df.height:
                actual = df["actual_home_score"] > df["actual_away_score"]
                pred = df["pred_margin"] > 0
                for hit, m in zip((pred == actual).to_list(), df["pred_margin"].abs().to_list()):
                    h = int(hit)
                    all_hits.append(h)
                    for t in (7, 10, 14):
                        if m >= t:
                            gate_hits[t].append(h)

        overall = _rate(all_hits) or {"n": 0, "winner_accuracy": 0.0}
        gated = {str(t): _rate(gate_hits[t]) for t in (7, 10, 14)}
        # Score: prioritize gate_10 accuracy, then gate_14, then overall, require sample
        g10 = gated["10"] or {"n": 0, "winner_accuracy": 0.0}
        g14 = gated["14"] or {"n": 0, "winner_accuracy": 0.0}
        score = (
            (g10.get("winner_accuracy") or 0) * 100
            + (g14.get("winner_accuracy") or 0) * 20
            + (overall.get("winner_accuracy") or 0) * 15
            + min(g10.get("n", 0), 400) / 400.0 * 3
        )
        # Prefer configs that clear 65% gated with decent n
        if (g10.get("n") or 0) >= 80 and (g10.get("winner_accuracy") or 0) >= 0.65:
            score += 8
        if (overall.get("winner_accuracy") or 0) >= 0.55:
            score += 2

        row = {
            "name": cfg["name"],
            "family": cfg["family"],
            "params": cfg["elo"],
            "overall": overall,
            "gated": gated,
            "score": round(score, 4),
            "seasons": season_rows,
        }
        results.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
        if (i + 1) % 15 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(uniq)}] best={best['name']} "
                f"g10={(best['gated'].get('10') or {}).get('winner_accuracy')} "
                f"overall={(best['overall'] or {}).get('winner_accuracy')}"
            )

    results.sort(key=lambda r: -r["score"])
    return {"best": best, "top": results[:20], "n_configs": len(uniq)}


def choose_gate(preds: pl.DataFrame, seasons: list[int]) -> dict:
    """
    Pick confidence gate among 7/10/14.

    Prefer |margin|>=10 when it clears >=65% with n>=150 (actionable CFB mismatches).
    Otherwise fall back to the best volume-aware gate that clears 65%.
    """
    import math

    candidates = []
    for t in (7.0, 10.0, 14.0):
        hits = []
        for season in seasons:
            sp = preds.filter(
                (pl.col("season") == season)
                & pl.col("actual_home_score").is_not_null()
                & (pl.col("both_fbs") == True)  # noqa: E712
            )
            if not sp.height:
                continue
            actual = sp["actual_home_score"] > sp["actual_away_score"]
            pred = sp["pred_margin"] > 0
            for hit, m in zip((pred == actual).to_list(), sp["pred_margin"].abs().to_list()):
                if m >= t:
                    hits.append(int(hit))
        rate = _rate(hits)
        if not rate:
            continue
        n = rate["n"]
        acc = rate["winner_accuracy"] or 0.0
        ok = 1 if acc >= 0.65 and n >= 80 else 0
        prefer10 = 1 if t == 10.0 and acc >= 0.65 and n >= 150 else 0
        score = acc * math.log10(max(n, 10))
        candidates.append(
            {
                "threshold": t,
                **rate,
                "ok": ok,
                "prefer10": prefer10,
                "score": round(score, 4),
            }
        )
    candidates.sort(key=lambda c: (c["prefer10"], c["ok"], c["score"]), reverse=True)
    return candidates[0] if candidates else {"threshold": 10.0, "n": 0, "winner_accuracy": None}


def main() -> None:
    ensure_dirs()
    # Prefer 2021-2024; include 2025 if present
    seasons = [2021, 2022, 2023, 2024, 2025]
    hist_start = 2018
    print(f"Loading schedules {hist_start}-{max(seasons)}…")
    sched = load_schedules(
        list(range(hist_start, max(seasons) + 1)),
        regular_only=True,
        fbs_only=True,
        include_cupcakes=True,
    )
    print(f"Games loaded: {sched.height} | source: {data_source_used()}")

    # Drop seasons with no completed both-FBS games
    usable = []
    for s in seasons:
        n = sched.filter(
            (pl.col("season") == s)
            & pl.col("home_score").is_not_null()
            & (pl.col("both_fbs") == True)  # noqa: E712
        ).height
        print(f"  season {s}: {n} completed both-FBS")
        if n >= 200:
            usable.append(s)
    if not usable:
        usable = seasons[:4]

    ablation = run_ablation(sched, usable)
    best = ablation["best"]
    print("Best config:", best["name"], best["params"], best["gated"])

    # Refit weights file from best
    weights = deepcopy(DEFAULT_WEIGHTS)
    weights["elo"].update(best["params"])
    # Restore situational weights for full family; zero extras for elo_only
    if best["family"] == "elo_only":
        weights["factors"]["recent_form"]["weight"] = 0.0
        weights["factors"]["rest"]["weight"] = 0.0
        weights["factors"]["conference_rivalry"]["weight"] = 0.0
        weights["factors"]["similar_opponent"]["weight"] = 0.0
        weights["elo"]["blend"] = 1.0
    elif best["family"] == "elo_form":
        weights["factors"]["rest"]["weight"] = 0.0
        weights["factors"]["conference_rivalry"]["weight"] = 0.0
        weights["factors"]["similar_opponent"]["weight"] = 0.0

    preds = predict_games_walkforward(sched, weights)
    gate_choice = choose_gate(preds, usable)
    gate = float(gate_choice["threshold"])
    weights["confidence"] = {
        "min_abs_margin": gate,
        "notes": (
            f"Walk-forward gated accuracy {gate_choice.get('winner_accuracy')} "
            f"on |margin|>={gate:g} (n={gate_choice.get('n')})"
        ),
    }
    weights["margin_sigma"] = 16.5
    weights["notes"] = (
        f"CFB audit chose {best['name']}: K={best['params']['k']}, "
        f"HFA_elo={best['params']['hfa_elo']}, blend={best['params']['blend']}, "
        f"gate={gate:g}. Injuries unused. Source: {data_source_used()}."
    )
    save_weights(weights)

    # Final per-season report with chosen params
    season_report = []
    all_hits: list[int] = []
    gated_hits: list[int] = []
    for season in usable:
        sp = preds.filter(pl.col("season") == season)
        bt = backtest_margins(sp, both_fbs_only=True)
        gates = gated_accuracy(sp, thresholds=[7, 10, 14], both_fbs_only=True)
        df = sp.filter(
            pl.col("actual_home_score").is_not_null()
            & (pl.col("both_fbs") == True)  # noqa: E712
        )
        if df.height:
            actual = df["actual_home_score"] > df["actual_away_score"]
            pred = df["pred_margin"] > 0
            for hit, m in zip((pred == actual).to_list(), df["pred_margin"].abs().to_list()):
                h = int(hit)
                all_hits.append(h)
                if m >= gate:
                    gated_hits.append(h)
        primary = {"threshold": gate, "n": 0, "winner_accuracy": None}
        if df.height:
            actual = (df["actual_home_score"] > df["actual_away_score"]).to_list()
            pred = (df["pred_margin"] > 0).to_list()
            margins = df["pred_margin"].abs().to_list()
            sel = [int(p == a) for p, a, m in zip(pred, actual, margins) if m >= gate]
            primary = {"threshold": gate, **(_rate(sel) or {"n": 0, "winner_accuracy": None})}
        season_report.append(
            {
                "season": season,
                **bt,
                "gates": gates,
                "primary_gate": primary,
            }
        )

    report = {
        "sport": "CFB",
        "data_source": data_source_used(),
        "seasons": usable,
        "chosen_params": {
            "k": best["params"]["k"],
            "hfa_elo": best["params"]["hfa_elo"],
            "elo_blend": best["params"]["blend"],
            "cupcake_k_scale": best["params"].get("cupcake_k_scale", 0.35),
            "margin_sigma": weights["margin_sigma"],
            "confidence_gate": gate,
            "family": best["family"],
            "config_name": best["name"],
        },
        "overall": _rate(all_hits),
        "gated": _rate(gated_hits),
        "gates_7_10_14": {
            "7": gated_accuracy(preds.filter(pl.col("season").is_in(usable)), [7], both_fbs_only=True).get("gate_7"),
            "10": gated_accuracy(preds.filter(pl.col("season").is_in(usable)), [10], both_fbs_only=True).get("gate_10"),
            "14": gated_accuracy(preds.filter(pl.col("season").is_in(usable)), [14], both_fbs_only=True).get("gate_14"),
        },
        "by_season": season_report,
        "ablation_top": ablation["top"][:12],
        "home_baseline_note": "Home teams win ~55% of FBS games historically; overall should beat that.",
    }

    out_path = ROOT / "output" / "cfb_audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")
    print("Overall:", report["overall"])
    print("Gated:", report["gated"])
    print("Params:", report["chosen_params"])


if __name__ == "__main__":
    main()
