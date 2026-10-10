"""
Explicit CFB factor weights.

Compared to NFL: higher Elo K, stronger HFA, wider margin sigma, no injuries.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from cfb_edge.config import WEIGHTS_PATH, ensure_dirs

DEFAULT_WEIGHTS = {
    "factors": {
        "elo": {
            "weight": 0.48,
            "description": "Pre-game team Elo (higher K than NFL; cupcake updates shrunk)",
        },
        "recent_form": {
            "weight": 0.22,
            "description": "Rolling points for/against (last 4 games)",
        },
        "home_field": {
            "weight": 0.14,
            "description": "CFB HFA historically larger than NFL; zeroed on neutral sites",
        },
        "rest": {
            "weight": 0.06,
            "description": "Rest / short week days between games when available",
        },
        "conference_rivalry": {
            "weight": 0.05,
            "description": "Conference game flag; mild rivalry/notes signal when present",
        },
        "similar_opponent": {
            "weight": 0.05,
            "description": "Light similar-opponent form (thinner than NFL deep matchups)",
        },
        "injuries": {
            "weight": 0.0,
            "description": "CFB injury feeds are sparse/unreliable; intentionally unused",
        },
    },
    "elo": {
        "k": 28.0,
        "hfa_elo": 65.0,
        "points_per_elo": 0.04,
        "cupcake_k_scale": 0.35,
        "season_regress": 0.25,
        "blend": 0.60,
    },
    "hfa_points": 3.0,
    "hfa_extra_points": 0.4,
    "form_games": 4,
    "form_scale": 0.28,
    "rest_pts_per_day": 0.18,
    "short_week_days": 6,
    "short_week_pts": 0.8,
    "rest_clamp": 2.2,
    "conf_game_hfa_dampen": 0.15,
    "rivalry_margin_bump": 0.4,
    "similar_scale": 0.12,
    "margin_sigma": 16.5,
    "total_sigma": 12.0,
    "league_total": 52.0,
    "edge_thresholds": {
        "totals_min_ev": 0.04,
        "spreads_min_ev": 0.05,
        "h2h_min_ev": 0.05,
    },
    "confidence": {
        "min_abs_margin": 10.0,
        "notes": "Gated winner picks when |pred_margin| clears threshold",
    },
    "notes": (
        "CFB: higher Elo K, stronger HFA, wider margin sigma vs NFL. "
        "Injuries weight=0 (no fake injury model). Cupcake Elo updates shrunk."
    ),
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
