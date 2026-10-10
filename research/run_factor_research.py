"""Historical factor research → research/factor_weights.json."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from nfl_edge.config import RESEARCH_DIR, WEIGHTS_PATH, ensure_dirs
from nfl_edge.features.weather import DEFAULT_WEIGHTS, load_weights, save_weights
from nfl_edge.ingest.nflverse import completed_games, load_schedules


def _hfa_by_era(games: pl.DataFrame) -> dict[str, float]:
    df = games.with_columns(
        margin=pl.col("home_score") - pl.col("away_score"),
        era=pl.when(pl.col("season") <= 2015)
        .then(pl.lit("pre_2016"))
        .when(pl.col("season") <= 2019)
        .then(pl.lit("2016_2019"))
        .when(pl.col("season") == 2020)
        .then(pl.lit("2020"))
        .otherwise(pl.lit("2021_plus")),
    )
    # exclude obvious neutrals if flagged
    if "location" in df.columns:
        df = df.filter(pl.col("location").is_null() | (pl.col("location") != "Neutral"))
    summary = df.group_by("era").agg(pl.col("margin").mean().alias("hfa"))
    out = {r["era"]: round(float(r["hfa"]), 2) for r in summary.iter_rows(named=True)}
    # ensure keys
    for k, v in DEFAULT_WEIGHTS["hfa_points"].items():
        out.setdefault(k, v)
    return out


def _weather_effects(games: pl.DataFrame) -> dict[str, Any]:
    """Estimate wind impact on totals from historical schedules."""
    df = games.filter(pl.col("home_score").is_not_null())
    df = df.with_columns(total=pl.col("home_score") + pl.col("away_score"))

    outdoor = df.filter(
        pl.col("roof").is_null()
        | pl.col("roof").cast(pl.Utf8).str.to_lowercase().is_in(["outdoors", "open", "outdoor"])
    )
    indoor = df.filter(
        pl.col("roof").cast(pl.Utf8).str.to_lowercase().is_in(["dome", "closed", "retractable"])
    )

    base_total = float(outdoor.filter(pl.col("wind").fill_null(0) <= 5)["total"].mean() or 45.0)
    high_wind = outdoor.filter(pl.col("wind").fill_null(0) >= 20)
    mid_wind = outdoor.filter((pl.col("wind").fill_null(0) >= 10) & (pl.col("wind") < 20))

    high_total = float(high_wind["total"].mean()) if high_wind.height > 30 else base_total - 6
    mid_total = float(mid_wind["total"].mean()) if mid_wind.height > 50 else base_total - 2.5
    indoor_total = float(indoor["total"].mean()) if indoor.height > 50 else base_total + 1.5

    # per 10mph approx from mid bucket (~12.5 mph avg vs ~2.5 calm)
    per_10 = (mid_total - base_total) / 1.0  # rough
    # clamp to literature-ish range
    per_10 = max(min(per_10, -1.0), -4.0)

    return {
        "wind_pass_yards_pct_per_10mph": -0.068,
        "wind_total_points_per_10mph": round(per_10, 2),
        "wind_high_threshold_mph": 20,
        "wind_high_total_extra": round((high_total - base_total) - per_10 * 2, 2),
        "rain_pass_yards_pct": -0.12,
        "rain_total_points": -2.0,
        "dome_total_boost": round(indoor_total - base_total, 2),
        "empirical": {
            "outdoor_calm_total": round(base_total, 2),
            "mid_wind_total": round(mid_total, 2),
            "high_wind_total": round(high_total, 2),
            "indoor_total": round(indoor_total, 2),
            "n_outdoor": outdoor.height,
            "n_high_wind": high_wind.height,
        },
    }


def _momentum_by_phase(games: pl.DataFrame) -> dict[str, Any]:
    """
    Measure how much last-3-game margin predicts next margin, by season phase.
    Returns recommended form weights (scaled correlation).
    """
    # build team game margins
    home = games.select(
        "season",
        "week",
        team=pl.col("home_team"),
        margin=pl.col("home_score") - pl.col("away_score"),
    )
    away = games.select(
        "season",
        "week",
        team=pl.col("away_team"),
        margin=pl.col("away_score") - pl.col("home_score"),
    )
    tg = pl.concat([home, away]).sort(["team", "season", "week"])
    tg = tg.with_columns(
        form=pl.col("margin").shift(1).rolling_mean(window_size=3, min_samples=1).over("team"),
        phase=pl.when(pl.col("week") <= 4)
        .then(pl.lit("early"))
        .when(pl.col("week") >= 15)
        .then(pl.lit("late"))
        .otherwise(pl.lit("mid")),
    ).drop_nulls(["form"])

    weights = {}
    for phase in ["early", "mid", "late"]:
        sub = tg.filter(pl.col("phase") == phase)
        if sub.height < 200:
            weights[phase] = {"corr": 0.0, "form_weight": DEFAULT_WEIGHTS["momentum"][f"form_weight_{phase}"]}
            continue
        # Pearson corr via covariance
        f = sub["form"].to_numpy()
        m = sub["margin"].to_numpy()
        f = f - f.mean()
        m = m - m.mean()
        denom = (f.std() * m.std()) or 1.0
        corr = float((f * m).mean() / denom)
        # map corr ~0.05-0.2 into form weights ~0.1-0.3
        fw = max(0.05, min(0.35, abs(corr) * 1.2))
        weights[phase] = {"corr": round(corr, 4), "form_weight": round(fw, 3)}

    return {
        "form_weight_early": weights["early"]["form_weight"],
        "form_weight_mid": weights["mid"]["form_weight"],
        "form_weight_late": weights["late"]["form_weight"],
        "early_weeks": 4,
        "late_week_start": 15,
        "empirical_corr": {k: v["corr"] for k, v in weights.items()},
    }


def _preseason_note(schedules_all: pl.DataFrame) -> dict[str, Any]:
    """Quick check: preseason margins vs regular — we still exclude preseason."""
    if "game_type" not in schedules_all.columns:
        return {"excluded": True, "reason": "no game_type column"}
    pre = schedules_all.filter(pl.col("game_type") == "PRE")
    pre = pre.filter(pl.col("home_score").is_not_null())
    if pre.is_empty():
        return {"excluded": True, "n": 0}
    # correlation of preseason win with regular season win% is out of scope; just note variance
    return {
        "excluded": True,
        "n_preseason_games": pre.height,
        "mean_total": round(float((pre["home_score"] + pre["away_score"]).mean()), 2),
        "note": "Preseason excluded from model; starters limited / variance high.",
    }


def run_research(start_season: int = 2006, end_season: int | None = None) -> dict:
    ensure_dirs()
    from nfl_edge.config import current_nfl_season

    seasons = list(range(start_season, (end_season or current_nfl_season()) + 1))
    # Use our cached REG loader for main analysis
    reg = completed_games(load_schedules(seasons, regular_only=True))

    # Only the research-owned keys are refit; tuned sections (elo, confidence, props,
    # deep_matchup, situational, market_blend, ...) from other audits are preserved.
    weights = {**DEFAULT_WEIGHTS, **load_weights(WEIGHTS_PATH)}
    weights["hfa_points"] = _hfa_by_era(reg)
    weather = _weather_effects(reg)
    empirical_wx = weather.pop("empirical", {})
    weights["weather"] = weather
    weights["momentum"] = _momentum_by_phase(reg)

    # keep injury / edge thresholds from defaults
    weights["research_meta"] = {
        "seasons": f"{start_season}-{seasons[-1]}",
        "n_reg_games": reg.height,
        "weather_empirical": empirical_wx,
        "preseason": _preseason_note(
            # lightweight: only current cache REG; preseason note may be sparse
            load_schedules(seasons[-3:], regular_only=False)
            if seasons
            else pl.DataFrame()
        ),
    }
    weights["notes"] = (
        "Weights fit from nflverse regular-season history. "
        "HFA is mean home margin by era. Weather totals from outdoor wind splits. "
        "Momentum weights from form→margin correlation by season phase."
    )

    path = save_weights(weights, WEIGHTS_PATH)
    # also write a human-readable summary
    summary_path = RESEARCH_DIR / "factor_summary.json"
    summary_path.write_text(json.dumps(weights.get("research_meta", {}), indent=2) + "\n")
    return {"path": str(path), "weights": weights}


if __name__ == "__main__":
    result = run_research()
    print(json.dumps({k: result["weights"][k] for k in ["hfa_points", "weather", "momentum"]}, indent=2))
    print("Wrote", result["path"])
