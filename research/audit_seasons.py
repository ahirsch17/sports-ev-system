"""Multi-season walk-forward audit for NFL + MLB with confidence gates."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from nfl_edge.config import OUTPUT_DIR, ensure_dirs
from nfl_edge.features.weather import load_weights
from nfl_edge.ingest.nflverse import load_schedules, load_team_stats
from nfl_edge.models.game import backtest_margins, predict_slate


def _rate(hits: list[int]) -> dict | None:
    if not hits:
        return None
    return {"n": len(hits), "winner_accuracy": round(sum(hits) / len(hits), 4)}


def audit_nfl(seasons: list[int] | None = None) -> dict:
    seasons = seasons or [2021, 2022, 2023, 2024, 2025]
    weights = load_weights()
    conf_min = float((weights.get("confidence") or {}).get("min_abs_margin", 5.0))
    rows = []
    all_hits: list[int] = []
    conf_hits: list[int] = []
    conf3_hits: list[int] = []
    conf7_hits: list[int] = []

    for season in seasons:
        hist = list(range(season - 2, season + 1))
        sched = load_schedules(hist)
        stats = load_team_stats(hist)
        preds = predict_slate(sched, stats, None, weights)
        season_preds = preds.filter(
            (pl.col("season") == season) & pl.col("actual_home_score").is_not_null()
        )
        bt = backtest_margins(season_preds)

        hits = []
        c_hits = []
        c3 = []
        c7 = []
        if season_preds.height:
            actual_home = season_preds["actual_home_score"] > season_preds["actual_away_score"]
            pred_home = season_preds["pred_margin"] > 0
            for i, hit in enumerate((pred_home == actual_home).to_list()):
                h = int(hit)
                hits.append(h)
                m = abs(float(season_preds["pred_margin"][i]))
                if m >= conf_min:
                    c_hits.append(h)
                if m >= 3:
                    c3.append(h)
                if m >= 7:
                    c7.append(h)

        all_hits.extend(hits)
        conf_hits.extend(c_hits)
        conf3_hits.extend(c3)
        conf7_hits.extend(c7)

        ats = None
        if "spread_line" in sched.columns and season_preds.height:
            joined = season_preds.join(
                sched.select(["game_id", "spread_line"]), on="game_id", how="left"
            ).filter(pl.col("spread_line").is_not_null())
            if joined.height:
                actual_margin = joined["actual_home_score"] - joined["actual_away_score"]
                ats = float(
                    ((joined["pred_margin"] > joined["spread_line"]) == (actual_margin > joined["spread_line"])).mean()
                )

        rows.append(
            {
                **bt,
                "season": season,
                "ats_vs_line": round(ats, 4) if ats is not None else None,
                "conf_gate": _rate(c_hits),
                "conf_ge_3": _rate(c3),
                "conf_ge_7": _rate(c7),
            }
        )

    overall_n = sum(r["n"] for r in rows)
    overall_acc = sum(r["n"] * r["winner_accuracy"] for r in rows) / max(overall_n, 1)
    return {
        "sport": "NFL",
        "seasons": rows,
        "overall_winner_accuracy": round(overall_acc, 4),
        "n": overall_n,
        "confidence": {
            "min_abs_margin": conf_min,
            "gated": _rate(conf_hits),
            "ge_3pts": _rate(conf3_hits),
            "ge_7pts": _rate(conf7_hits),
        },
    }


def audit_mlb(days_back: int = 180) -> dict:
    from mlb_edge.elo import build_mlb_elo, elo_run_expectation
    from mlb_edge.ingest import completed_games, fetch_pitcher_stats, schedule_frame
    from mlb_edge.predict import _park_factor, _rolling_form
    from mlb_edge.scoring import predict_runs
    from mlb_edge.weights import load_weights as mlb_weights

    weights = mlb_weights()
    conf = weights.get("confidence") or {}
    min_margin = float(conf.get("min_abs_margin", 0.5))
    min_sp_gap = float(conf.get("min_sp_era_gap", 1.0))
    elo_k = float(weights.get("elo_k", 6.0))
    form_w = float(weights.get("form_blend", 0.45))
    elo_w = float(weights.get("elo_blend", 0.45))
    league_w = max(0.0, 1.0 - form_w - elo_w)

    end = date.today()
    start = end - timedelta(days=days_back)
    done = completed_games(schedule_frame(start, end)).sort("gameday")
    season = int(done["season"].max()) if done.height and "season" in done.columns else end.year

    pitcher_cache: dict[int, dict] = {}
    flat: list[int] = []
    gated: list[int] = []
    sp_gap_hits: list[int] = []
    by_month: dict[str, list[int]] = {}

    days = done["gameday"].unique().sort().to_list()
    for day in days:
        prior = done.filter(pl.col("gameday") < day)
        if prior.height < 120:
            continue
        form = _rolling_form(prior, window=15)
        elo = build_mlb_elo(prior, k=elo_k)
        for r in done.filter(pl.col("gameday") == day).iter_rows(named=True):
            hs, aws = float(r["home_score"]), float(r["away_score"])
            if hs == aws:
                continue
            home = str(r["home_team"]).upper()
            away = str(r["away_team"]).upper()
            home_sp: dict = {}
            away_sp: dict = {}
            if r.get("home_sp_id"):
                home_sp = pitcher_cache.setdefault(
                    int(r["home_sp_id"]), fetch_pitcher_stats(int(r["home_sp_id"]), season)
                )
            if r.get("away_sp_id"):
                away_sp = pitcher_cache.setdefault(
                    int(r["away_sp_id"]), fetch_pitcher_stats(int(r["away_sp_id"]), season)
                )
            if not home_sp or not away_sp:
                continue

            hf = float(form.get(home, {}).get("rpg", 4.5) or 4.5)
            af = float(form.get(away, {}).get("rpg", 4.5) or 4.5)
            eh, ea = elo.get(home, 1500.0), elo.get(away, 1500.0)
            home_rpg = form_w * hf + elo_w * elo_run_expectation(eh, ea) + league_w * 4.5
            away_rpg = form_w * af + elo_w * elo_run_expectation(ea, eh) + league_w * 4.5
            scored = predict_runs(
                home_rpg=home_rpg,
                away_rpg=away_rpg,
                home_sp=home_sp,
                away_sp=away_sp,
                park_factor=_park_factor(home, weights),
                temp=r.get("temp"),
                wind=r.get("wind"),
                weights=weights,
            )
            hit = int((scored["pred_margin"] > 0) == (hs > aws))
            flat.append(hit)
            era_gap = abs(float(home_sp.get("era") or 4.5) - float(away_sp.get("era") or 4.5))
            if era_gap >= min_sp_gap:
                sp_gap_hits.append(hit)
            if era_gap >= min_sp_gap and abs(scored["pred_margin"]) >= min_margin:
                gated.append(hit)
            by_month.setdefault(str(day)[:7], []).append(hit)

    months = {m: round(sum(v) / len(v), 4) for m, v in by_month.items()}
    return {
        "sport": "MLB",
        "n": len(flat),
        "winner_accuracy": round(sum(flat) / len(flat), 4) if flat else None,
        "home_baseline": round(float((done["home_score"] > done["away_score"]).mean()), 4) if done.height else None,
        "by_month": months,
        "confidence": {
            "min_abs_margin": min_margin,
            "min_sp_era_gap": min_sp_gap,
            "sp_gap_only": _rate(sp_gap_hits),
            "gated": _rate(gated),
        },
        "notes": "form+Elo+SP walk-forward; confidence = SP ERA gap + margin gate",
    }


def main():
    ensure_dirs()
    print("=== NFL audit ===")
    nfl = audit_nfl()
    print(json.dumps(nfl, indent=2))
    print("=== MLB audit ===")
    mlb = audit_mlb()
    print(json.dumps(mlb, indent=2))
    out = OUTPUT_DIR / "audit_report.json"
    out.write_text(json.dumps({"nfl": nfl, "mlb": mlb}, indent=2) + "\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
