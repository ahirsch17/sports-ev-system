"""Shared MLB game scoring — fixed SP/bullpen signs, walk-forward friendly."""

from __future__ import annotations

from typing import Any

from mlb_edge.weights import factor_list, load_weights


def sp_run_delta(sp: dict) -> float:
    """
    Expected runs allowed delta vs league average (~4.5 RPG against).
    Negative = good pitcher (suppresses opponent scoring).
    """
    if not sp:
        return 0.0
    era = float(sp.get("era") or 4.50)
    whip = float(sp.get("whip") or 1.35)
    k9 = float(sp.get("k_per_9") or 8.0)
    bb9 = float(sp.get("bb_per_9") or 3.2)
    ip = float(sp.get("ip") or 0.0)
    # quality index centered at league
    q = (era - 4.20) * 0.65 + (whip - 1.30) * 1.1 - (k9 - 8.5) * 0.07 + (bb9 - 3.1) * 0.10
    # shrink when tiny sample
    shrink = min(1.0, ip / 40.0) if ip > 0 else 0.35
    return q * (5.5 / 9.0) * max(0.35, shrink)


def predict_runs(
    *,
    home_rpg: float,
    away_rpg: float,
    home_sp: dict,
    away_sp: dict,
    park_factor: float = 1.0,
    temp: float | None = None,
    wind: str | None = None,
    home_bp_era: float = 4.2,
    away_bp_era: float = 4.2,
    weights: dict | None = None,
) -> dict[str, Any]:
    """
    Expected home/away runs with correctly signed pitcher effects.

    home_runs ≈ home_offense + away_SP_delta + park/weather/HFA share
    (good away SP → negative delta → fewer home runs)
    """
    weights = weights or load_weights()
    factors = {f["key"]: f["weight"] for f in factor_list(weights)}

    away_sp_d = sp_run_delta(away_sp)  # applied to HOME offense
    home_sp_d = sp_run_delta(home_sp)  # applied to AWAY offense

    # bullpen: high ERA → opponent scores more late (positive for opponent)
    away_bp_d = (away_bp_era - 4.10) * 0.18
    home_bp_d = (home_bp_era - 4.10) * 0.18

    w_sp = factors["starting_pitcher"]
    w_off = factors["offense_form"]
    w_bp = factors["bullpen"]
    w_park = factors["park_factor"]
    w_hfa = factors["home_field"]
    w_wx = factors["weather"]
    w_plt = factors["platoon_matchup"]
    w_sim = factors["similar_pitcher_history"]

    # Platoon proxy from SP handedness vs generic — small
    plt_home = 0.05 if (away_sp.get("throws") == "L") else 0.0
    plt_away = 0.05 if (home_sp.get("throws") == "L") else 0.0

    # Similar-SP: mild — good offense vs bad SP
    sim_home = max(-0.35, min(0.35, (home_rpg - 4.5) * 0.08 + (float(away_sp.get("era") or 4.5) - 4.2) * 0.1))
    sim_away = max(-0.35, min(0.35, (away_rpg - 4.5) * 0.08 + (float(home_sp.get("era") or 4.5) - 4.2) * 0.1))

    # Core: offense level + SP run delta (SIGNED correctly: +delta means more runs allowed)
    home_core = (
        w_off * home_rpg
        + w_sp * (home_rpg + away_sp_d)
        + w_bp * (home_rpg + away_bp_d)
        + w_plt * (home_rpg + plt_home)
        + w_sim * (home_rpg + sim_home)
    ) / max(w_off + w_sp + w_bp + w_plt + w_sim, 1e-6)

    away_core = (
        w_off * away_rpg
        + w_sp * (away_rpg + home_sp_d)
        + w_bp * (away_rpg + home_bp_d)
        + w_plt * (away_rpg + plt_away)
        + w_sim * (away_rpg + sim_away)
    ) / max(w_off + w_sp + w_bp + w_plt + w_sim, 1e-6)

    park_mult = 1.0 + w_park * (park_factor - 1.0) * 1.6
    wx = 0.0
    if temp is not None:
        if temp >= 85:
            wx += 0.2
        elif temp <= 50:
            wx -= 0.25
    if wind:
        wl = str(wind).lower()
        if "out" in wl:
            wx += 0.3
        elif "in" in wl:
            wx -= 0.25
    wx *= w_wx * 2.0

    hfa_runs = w_hfa * 1.85  # weight 0.07 → ~0.13 runs; scales with configured HFA weight
    # rest_travel: no schedule rest feed in slate yet; leave effect at 0 (weight documented as thin)

    home_runs = max(1.8, home_core * park_mult + wx / 2 + hfa_runs)
    away_runs = max(1.8, away_core * park_mult + wx / 2)

    return {
        "pred_home_runs": home_runs,
        "pred_away_runs": away_runs,
        "pred_margin": home_runs - away_runs,
        "pred_total": home_runs + away_runs,
        "away_sp_delta": away_sp_d,
        "home_sp_delta": home_sp_d,
        "breakdown": [
            {"factor": "starting_pitcher", "weight": w_sp, "home_effect": round(away_sp_d, 2), "away_effect": round(home_sp_d, 2),
             "detail": f"Away SP ERA {away_sp.get('era', 'n/a')} -> d{away_sp_d:+.2f} home runs | Home SP ERA {home_sp.get('era', 'n/a')} -> d{home_sp_d:+.2f}"},
            {"factor": "offense_form", "weight": w_off, "home_effect": round(home_rpg - 4.5, 2), "away_effect": round(away_rpg - 4.5, 2),
             "detail": f"Home RPG {home_rpg:.2f} | Away RPG {away_rpg:.2f}"},
            {"factor": "bullpen", "weight": w_bp, "home_effect": round(away_bp_d, 2), "away_effect": round(home_bp_d, 2),
             "detail": f"Opp BP ERA home faces {away_bp_era:.2f} / away faces {home_bp_era:.2f}"},
            {"factor": "park_factor", "weight": w_park, "home_effect": round((park_factor - 1) * 2, 2), "away_effect": round((park_factor - 1) * 2, 2),
             "detail": f"Park {park_factor*100:.0f}"},
            {"factor": "home_field", "weight": w_hfa, "home_effect": round(hfa_runs, 2), "away_effect": 0.0, "detail": f"HFA {hfa_runs:.2f} runs (weight {w_hfa})"},
            {"factor": "weather", "weight": w_wx, "home_effect": round(wx / 2, 2), "away_effect": round(wx / 2, 2),
             "detail": f"temp={temp} wind={wind}"},
            {"factor": "platoon_matchup", "weight": w_plt, "home_effect": round(plt_home, 2), "away_effect": round(plt_away, 2),
             "detail": f"vs {away_sp.get('throws') or '?'} / vs {home_sp.get('throws') or '?'}"},
            {"factor": "similar_pitcher_history", "weight": w_sim, "home_effect": round(sim_home, 2), "away_effect": round(sim_away, 2),
             "detail": "Offense vs SP quality band"},
            {"factor": "rest_travel", "weight": factors["rest_travel"], "home_effect": 0.0, "away_effect": 0.0, "detail": "Unused (no rest feed)"},
        ],
    }
