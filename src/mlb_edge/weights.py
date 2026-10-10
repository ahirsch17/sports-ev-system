"""
Explicit MLB factor weights — sum ≈ 1.0 for run-expectancy contributions.
These drive both the model and the UI breakdown.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mlb_edge.config import WEIGHTS_PATH, ensure_dirs

DEFAULT_WEIGHTS = {
    "factors": {
        # Starting pitcher quality dominates 1-game MLB outcomes
        "starting_pitcher": {
            "weight": 0.28,
            "description": "SP recent ERA / K-BB / IP — largest single-game lever",
        },
        "platoon_matchup": {
            "weight": 0.12,
            "description": "Lineup vs RHP/LHP splits vs starter handedness",
        },
        "offense_form": {
            "weight": 0.14,
            "description": "Rolling team runs / OPS-ish proxy last 10–15 games",
        },
        "bullpen": {
            "weight": 0.12,
            "description": "Bullpen ERA / leverage usage last 7–14 days",
        },
        "park_factor": {
            "weight": 0.10,
            "description": "Ballpark run environment (Coors ↑, pitchers' parks ↓)",
        },
        "home_field": {
            "weight": 0.06,
            "description": "Historical MLB HFA ~54% / ~0.1–0.15 runs",
        },
        "weather": {
            "weight": 0.08,
            "description": "Temp + wind out/in — mostly totals, some run impact",
        },
        "rest_travel": {
            "weight": 0.0,
            "description": "Off-days / travel (unused until rest feed is wired)",
        },
        "similar_pitcher_history": {
            "weight": 0.06,
            "description": "Offense vs similar SP profiles, weighted by lineup continuity",
        },
    },
    "run_sigma": 3.2,
    "edge_thresholds": {
        "totals_min_ev": 0.04,
        "spreads_min_ev": 0.05,
        "h2h_min_ev": 0.045,
        "props_min_ev": 0.03,
    },
    "park_run_factors": {
        # Approximate multi-year run factors (100 = average). Editable.
        "COL": 118,
        "CIN": 108,
        "BOS": 105,
        "TEX": 104,
        "PHI": 103,
        "CHC": 102,
        "NYY": 102,
        "LAD": 101,
        "ATL": 101,
        "HOU": 100,
        "TOR": 100,
        "MIN": 100,
        "MIL": 99,
        "NYM": 98,
        "SF": 94,
        "SD": 93,
        "SEA": 92,
        "ATH": 96,
        "OAK": 96,
        "TB": 95,
        "MIA": 94,
        "DET": 97,
        "CLE": 98,
        "CWS": 99,
        "CHW": 99,
        "KC": 101,
        "STL": 99,
        "ARI": 102,
        "AZ": 102,
        "WSH": 100,
        "WAS": 100,
        "BAL": 103,
        "PIT": 97,
        "LAA": 100,
    },
    "notes": "MLB weights prioritize starting pitcher + platoon; park/weather mainly move totals.",
}


def load_weights(path: Path | None = None) -> dict:
    path = path or WEIGHTS_PATH
    if path.exists():
        return json.loads(path.read_text())
    return deepcopy(DEFAULT_WEIGHTS)


def save_weights(weights: dict, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or WEIGHTS_PATH
    path.write_text(json.dumps(weights, indent=2) + "\n")
    return path


def factor_list(weights: dict | None = None) -> list[dict]:
    w = weights or load_weights()
    rows = []
    for key, meta in w["factors"].items():
        rows.append(
            {
                "key": key,
                "weight": float(meta["weight"]),
                "description": meta.get("description", ""),
            }
        )
    rows.sort(key=lambda r: -r["weight"])
    return rows
