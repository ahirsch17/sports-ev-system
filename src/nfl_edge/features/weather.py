"""Weather and situational feature helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nfl_edge.config import WEIGHTS_PATH


DEFAULT_WEIGHTS = {
    "hfa_points": {
        "pre_2016": 2.6,
        "2016_2019": 1.8,
        "2020": 0.2,
        "2021_plus": 1.4,
    },
    "weather": {
        "wind_pass_yards_pct_per_10mph": -0.068,
        "wind_total_points_per_10mph": -2.5,
        "wind_high_threshold_mph": 20,
        "wind_high_total_extra": -4.0,
        "rain_pass_yards_pct": -0.12,
        "rain_total_points": -2.0,
        "dome_total_boost": 1.5,
    },
    "momentum": {
        "form_weight_early": 0.15,
        "form_weight_mid": 0.25,
        "form_weight_late": 0.20,
        "early_weeks": 4,
        "late_week_start": 15,
    },
    "injuries": {
        "qb1_out_points": -4.5,
        "qb2_out_points": -1.5,
        "wr1_out_points": -1.2,
        "rb1_out_points": -0.8,
        "ol_starter_out_points": -0.6,
        "cb1_out_vs_pass": 1.0,
        "inactive_depth_gt_2": 0.0,
    },
    "edge_thresholds": {
        "props_min_ev": 0.03,
        "totals_min_ev": 0.04,
        "spreads_min_ev": 0.05,
        "h2h_min_ev": 0.05,
    },
    "deep_matchup": {
        "weight": 0.42,
        "top_k_similar": 8,
        "max_style_distance": 1.35,
        "qb1_continuity_weight": 0.40,
        "sim_scale": 0.55,
        "mismatch_scale": 0.40,
        "sim_margin_scale": 0.16,
    },
    "situational": {
        "weight": 0.55,
        "rest_pts_per_day": 0.15,
        "short_week_days": 5,
        "short_week_pts": 0.6,
        "thursday_hfa_dampen": 0.25,
        "rest_clamp": 2.0,
        "division_hfa_dampen": 0.2,
        "division_clamp": 0.5,
        "pace_pass_scale": 0.7,
        "pace_rush_scale": 0.4,
        "pace_clamp": 1.0,
    },
    "margin_sigma": 13.5,
    "total_sigma": 13.0,
    "notes": "Seed weights from literature; overwritten by `nfl-edge research`.",
}


def load_weights(path: Path | None = None) -> dict:
    path = path or WEIGHTS_PATH
    if path.exists():
        return json.loads(path.read_text())
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict, path: Path | None = None) -> Path:
    path = path or WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2) + "\n")
    return path


def hfa_for_season(season: int, weights: dict | None = None) -> float:
    w = (weights or load_weights())["hfa_points"]
    if season <= 2015:
        return float(w["pre_2016"])
    if season <= 2019:
        return float(w["2016_2019"])
    if season == 2020:
        return float(w["2020"])
    return float(w["2021_plus"])


def is_outdoor(roof: str | None) -> bool:
    if roof is None:
        return True
    return str(roof).lower() in {"outdoors", "open", "outdoor"}


def weather_adjustments(
    *,
    roof: str | None,
    temp: float | None,
    wind: float | None,
    weights: dict | None = None,
) -> dict[str, float]:
    """
    Return additive adjustments:
      total_points, pass_yards_pct, rush_yards_pct
    """
    w = (weights or load_weights())["weather"]
    if not is_outdoor(roof):
        return {
            "total_points": float(w.get("dome_total_boost", 1.5)),
            "pass_yards_pct": 0.02,
            "rush_yards_pct": 0.0,
        }

    wind_v = float(wind or 0.0)
    total = (wind_v / 10.0) * float(w["wind_total_points_per_10mph"])
    pass_pct = (wind_v / 10.0) * float(w["wind_pass_yards_pct_per_10mph"])
    rush_pct = -0.5 * pass_pct  # rushing share rises when passing falls

    if wind_v >= float(w["wind_high_threshold_mph"]):
        total += float(w["wind_high_total_extra"])

    # cold nudge
    if temp is not None and temp < 32:
        total -= 1.0
        pass_pct -= 0.03

    return {
        "total_points": total,
        "pass_yards_pct": pass_pct,
        "rush_yards_pct": rush_pct,
    }


def form_weight_for_week(week: int, weights: dict | None = None) -> float:
    m = (weights or load_weights())["momentum"]
    if week <= int(m["early_weeks"]):
        return float(m["form_weight_early"])
    if week >= int(m["late_week_start"]):
        return float(m["form_weight_late"])
    return float(m["form_weight_mid"])


def enrich_schedule_weather(schedules: pl.DataFrame, weights: dict | None = None) -> pl.DataFrame:
    """Add weather adjustment columns to schedule rows."""
    weights = weights or load_weights()
    rows = []
    for r in schedules.iter_rows(named=True):
        adj = weather_adjustments(
            roof=r.get("roof"),
            temp=r.get("temp"),
            wind=r.get("wind"),
            weights=weights,
        )
        rows.append(
            {
                "game_id": r.get("game_id"),
                "wx_total_adj": adj["total_points"],
                "wx_pass_pct": adj["pass_yards_pct"],
                "wx_rush_pct": adj["rush_yards_pct"],
                "hfa_points": hfa_for_season(int(r.get("season") or 2024), weights),
            }
        )
    if not rows:
        return schedules
    wx = pl.DataFrame(rows)
    return schedules.join(wx, on="game_id", how="left")
