"""
Measurable situational proxies (no vibes):
- Rest / short week / Thursday
- Division games (HFA dampener fit from history)
- Offense pace + pass-rate vs opponent defensive style (play-calling proxy)
"""

from __future__ import annotations

import polars as pl

from nfl_edge.features.deep_matchups import StyleVector, style_before


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rest_short_week_adj(
    *,
    home_rest: float,
    away_rest: float,
    weekday: str | None,
    weights: dict | None = None,
) -> dict[str, float | str]:
    """
    Points margin adj (home perspective) from rest differential + short-week flags.

    Empirically (2016-2025 REG): ~0.15-0.25 pts per day of rest edge when |diff|>=3;
    Thursday / mutual short weeks slightly compress HFA rather than flip favorites.
    """
    cfg = (weights or {}).get("situational") or {}
    per_day = float(cfg.get("rest_pts_per_day", 0.18))
    short_thresh = float(cfg.get("short_week_days", 5))
    short_penalty = float(cfg.get("short_week_pts", 0.9))
    thursday_hfa_dampen = float(cfg.get("thursday_hfa_dampen", 0.35))
    clamp_abs = float(cfg.get("rest_clamp", 2.5))

    rest_diff = float(home_rest) - float(away_rest)
    # Mild nonlinear: ignore 0-1 day noise, scale the rest
    if abs(rest_diff) <= 1:
        rest_pts = 0.0
    else:
        # discount the first day of gap
        signed = rest_diff - (1.0 if rest_diff > 0 else -1.0)
        rest_pts = signed * per_day

    home_short = float(home_rest) <= short_thresh
    away_short = float(away_rest) <= short_thresh
    short_pts = 0.0
    note_bits: list[str] = []

    if home_short and not away_short:
        short_pts -= short_penalty
        note_bits.append("home short week")
    elif away_short and not home_short:
        short_pts += short_penalty
        note_bits.append("away short week")
    elif home_short and away_short:
        note_bits.append("mutual short week")

    is_thursday = str(weekday or "").lower().startswith("thu")
    thursday_pts = 0.0
    if is_thursday:
        # Both usually short on TNF; slight home-field compression
        thursday_pts = -thursday_hfa_dampen
        note_bits.append("Thursday")

    margin_adj = _clamp(rest_pts + short_pts + thursday_pts, -clamp_abs, clamp_abs)
    if abs(rest_diff) >= 2 and "rest" not in " ".join(note_bits):
        note_bits.append(f"rest {home_rest:.0f}/{away_rest:.0f}")

    return {
        "rest_margin_adj": round(margin_adj, 3),
        "rest_diff": rest_diff,
        "home_short_week": 1.0 if home_short else 0.0,
        "away_short_week": 1.0 if away_short else 0.0,
        "is_thursday": 1.0 if is_thursday else 0.0,
        "rest_note": "; ".join(note_bits) if note_bits else "normal rest",
    }


def division_game_adj(
    *,
    is_div: bool,
    weights: dict | None = None,
) -> dict[str, float | str]:
    """
    Division games: slightly lower mean home margin historically (~0.3 pts).
    Modeled as a small HFA dampener (home perspective negative when div=1).
    """
    cfg = (weights or {}).get("situational") or {}
    dampen = float(cfg.get("division_hfa_dampen", 0.28))
    clamp_abs = float(cfg.get("division_clamp", 0.6))
    if not is_div:
        return {
            "div_margin_adj": 0.0,
            "is_division": 0.0,
            "div_note": "non-division",
        }
    adj = _clamp(-dampen, -clamp_abs, clamp_abs)
    return {
        "div_margin_adj": round(adj, 3),
        "is_division": 1.0,
        "div_note": "division game (HFA dampened)",
    }


def pace_pass_tendency_adj(
    *,
    offense: StyleVector,
    defense: StyleVector,
    weights: dict | None = None,
) -> dict[str, float | str]:
    """
    Play-calling proxy from rolling team styles:
    - High pass-rate offense vs high sack defense → points drag
    - High pass-rate offense vs soft sack / pass-friendly → points bump
    - Rush-heavy offense vs pass-rush-oriented D → slight rush edge
    Clamped hard so it cannot dominate Elo/points.
    """
    cfg = (weights or {}).get("situational") or {}
    pass_scale = float(cfg.get("pace_pass_scale", 1.4))
    rush_scale = float(cfg.get("pace_rush_scale", 0.8))
    clamp_abs = float(cfg.get("pace_clamp", 2.0))

    pass_rate_z = (offense.pass_rate - 0.55) / 0.08  # ~1 sd
    sack_z = (defense.def_sacks - 2.2) / 1.2
    # offense pass EPA already in style; light touch
    pass_epa_z = offense.pass_epa / 8.0
    rush_epa_z = offense.rush_epa / 5.0

    # mismatch: pass-happy vs elite pressure hurts; vs soft pressure helps
    pass_mismatch = pass_rate_z * (-0.55 * sack_z) + 0.35 * pass_epa_z * (1.0 - 0.4 * max(0.0, sack_z))
    rush_mismatch = (-pass_rate_z) * (0.25 * sack_z) + 0.3 * rush_epa_z

    points = pass_mismatch * pass_scale + rush_mismatch * rush_scale
    points = _clamp(points, -clamp_abs, clamp_abs)

    pass_pct = _clamp(pass_mismatch * 0.025, -0.08, 0.08)
    rush_pct = _clamp(rush_mismatch * 0.03, -0.08, 0.08)

    notes = []
    if pass_rate_z > 0.6 and sack_z > 0.6:
        notes.append("pass-heavy vs pressure")
    elif pass_rate_z > 0.6 and sack_z < -0.4:
        notes.append("pass-heavy vs soft rush")
    elif pass_rate_z < -0.6:
        notes.append("run-leaning scheme")
    if abs(points) < 0.15:
        notes = ["neutral scheme"]

    # pace proxy: more plays ~ slightly higher totals (applied later as half to each side)
    pace_total = _clamp((offense.plays - 62.0) / 8.0, -0.8, 0.8)

    return {
        "pace_points": round(points, 3),
        "pace_pass_pct": round(pass_pct, 4),
        "pace_rush_pct": round(rush_pct, 4),
        "pace_total_adj": round(pace_total, 3),
        "pace_note": "|".join(notes) if notes else "neutral scheme",
    }


def near_miss_form_adj(
    *,
    schedules: pl.DataFrame,
    team: str,
    season: int,
    week: int,
    weights: dict | None = None,
) -> dict[str, float | str]:
    """
    Credit a close loss (1-7 pts) as competitive form — e.g. TEN nearly beating PHI.
    Measurable only: prior final score margin. Clamped small.
    """
    cfg = (weights or {}).get("situational") or {}
    max_boost = float(cfg.get("near_miss_pts", 2.0))
    max_loss = float(cfg.get("near_miss_max_loss", 7.0))
    prior = schedules.filter(
        (
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    if prior.is_empty():
        return {"near_miss_pts": 0.0, "near_miss_note": ""}
    last = prior.sort(["season", "week"]).tail(1).to_dicts()[0]
    home = str(last["home_team"])
    away = str(last["away_team"])
    hs = float(last["home_score"])
    aws = float(last["away_score"])
    if team == home:
        margin = hs - aws  # positive = win
        opp = away
    else:
        margin = aws - hs
        opp = home
    if margin >= 0:
        return {"near_miss_pts": 0.0, "near_miss_note": ""}
    loss_by = abs(margin)
    if loss_by < 1 or loss_by > max_loss:
        return {"near_miss_pts": 0.0, "near_miss_note": ""}
    # Closer losses get more credit
    boost = max_boost * (1.0 - (loss_by - 1.0) / max_loss)
    boost = _clamp(boost, 0.5, max_boost)
    return {
        "near_miss_pts": round(boost, 3),
        "near_miss_note": f"close loss by {loss_by:.0f} vs {opp}",
    }


def build_situational_features(
    schedules: pl.DataFrame,
    styles: pl.DataFrame | None = None,
    weights: dict | None = None,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-game rest/division/pace columns for the matchup engine."""
    styles = styles if styles is not None else pl.DataFrame()
    # Full history needed for near-miss lookback (slate often passes upcoming-only)
    hist = history if history is not None and not history.is_empty() else schedules
    rows = []
    for r in schedules.iter_rows(named=True):
        season = int(r["season"])
        week = int(r["week"])
        home, away = str(r["home_team"]), str(r["away_team"])
        home_rest = float(r.get("home_rest") if r.get("home_rest") is not None else 7)
        away_rest = float(r.get("away_rest") if r.get("away_rest") is not None else 7)
        weekday = r.get("weekday")
        is_div = bool(int(r.get("div_game") or 0)) if r.get("div_game") is not None else False

        rest = rest_short_week_adj(
            home_rest=home_rest,
            away_rest=away_rest,
            weekday=str(weekday) if weekday is not None else None,
            weights=weights,
        )
        div = division_game_adj(is_div=is_div, weights=weights)

        home_off = style_before(styles, home, season, week) if not styles.is_empty() else StyleVector()
        away_off = style_before(styles, away, season, week) if not styles.is_empty() else StyleVector()
        home_def = style_before(styles, home, season, week) if not styles.is_empty() else StyleVector()
        away_def = style_before(styles, away, season, week) if not styles.is_empty() else StyleVector()

        home_pace = pace_pass_tendency_adj(offense=home_off, defense=away_def, weights=weights)
        away_pace = pace_pass_tendency_adj(offense=away_off, defense=home_def, weights=weights)

        home_nm = near_miss_form_adj(schedules=hist, team=home, season=season, week=week, weights=weights)
        away_nm = near_miss_form_adj(schedules=hist, team=away, season=season, week=week, weights=weights)
        near_miss_net = float(home_nm["near_miss_pts"]) - float(away_nm["near_miss_pts"])
        notes_clean = []
        if home_nm["near_miss_pts"]:
            notes_clean.append(f"home {home_nm['near_miss_note']}")
        if away_nm["near_miss_pts"]:
            notes_clean.append(f"away {away_nm['near_miss_note']}")

        # Net margin: rest + div + home offense pace - away offense pace + near-miss
        net = (
            float(rest["rest_margin_adj"])
            + float(div["div_margin_adj"])
            + float(home_pace["pace_points"])
            - float(away_pace["pace_points"])
            + near_miss_net
        )
        net = _clamp(net, -4.5, 4.5)
        total_pace = (float(home_pace["pace_total_adj"]) + float(away_pace["pace_total_adj"])) / 2.0

        rows.append(
            {
                "game_id": r["game_id"],
                "rest_margin_adj": rest["rest_margin_adj"],
                "rest_diff": rest["rest_diff"],
                "home_short_week": rest["home_short_week"],
                "away_short_week": rest["away_short_week"],
                "is_thursday": rest["is_thursday"],
                "rest_note": rest["rest_note"],
                "div_margin_adj": div["div_margin_adj"],
                "is_division": div["is_division"],
                "div_note": div["div_note"],
                "home_pace_points": home_pace["pace_points"],
                "away_pace_points": away_pace["pace_points"],
                "home_pace_pass_pct": home_pace["pace_pass_pct"],
                "away_pace_pass_pct": away_pace["pace_pass_pct"],
                "home_pace_rush_pct": home_pace["pace_rush_pct"],
                "away_pace_rush_pct": away_pace["pace_rush_pct"],
                "pace_total_adj": round(total_pace, 3),
                "home_pace_note": home_pace["pace_note"],
                "away_pace_note": away_pace["pace_note"],
                "near_miss_margin_adj": round(near_miss_net, 3),
                "near_miss_note": "; ".join(notes_clean) if notes_clean else "",
                "situational_net_margin": round(net, 3),
            }
        )

    if not rows:
        return pl.DataFrame({"game_id": []})
    return pl.DataFrame(rows)
