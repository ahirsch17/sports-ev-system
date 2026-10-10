"""CFB game scoring helpers — walk-forward friendly margin blend."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl

from cfb_edge.weights import factor_list, load_weights


def _parse_day(raw: Any) -> date | None:
    if raw is None:
        return None
    s = str(raw)[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except ValueError:
            return None


def rolling_form_state(completed: pl.DataFrame, window: int = 4) -> dict[str, dict]:
    """Latest rolling PF/PA per team from completed games only."""
    if completed.is_empty():
        return {}
    history: dict[str, list[tuple[float, float]]] = {}
    games = completed.sort(["season", "week", "gameday", "game_id"])
    for r in games.iter_rows(named=True):
        home, away = str(r["home_team"]), str(r["away_team"])
        hs, aws = float(r["home_score"]), float(r["away_score"])
        history.setdefault(home, []).append((hs, aws))
        history.setdefault(away, []).append((aws, hs))

    out: dict[str, dict] = {}
    for team, rows in history.items():
        tail = rows[-window:]
        pf = sum(x[0] for x in tail) / len(tail)
        pa = sum(x[1] for x in tail) / len(tail)
        out[team] = {"pf": pf, "pa": pa, "n": len(tail), "diff": pf - pa}
    return out


def rest_days_map(schedules: pl.DataFrame) -> dict[tuple[Any, str], float | None]:
    """Map (game_id, team) -> days since that team's previous game."""
    if schedules.is_empty():
        return {}
    last_day: dict[str, date] = {}
    out: dict[tuple[Any, str], float | None] = {}
    games = schedules.sort(["season", "week", "gameday", "game_id"])
    for r in games.iter_rows(named=True):
        day = _parse_day(r.get("gameday") or r.get("start_date"))
        gid = r.get("game_id")
        for team in (str(r["home_team"]), str(r["away_team"])):
            prev = last_day.get(team)
            if day is None or prev is None:
                out[(gid, team)] = None
            else:
                out[(gid, team)] = float((day - prev).days)
            if day is not None:
                last_day[team] = day
    return out


def similar_opp_signal(
    opp_elo: float,
    team_hist: list[dict],
    *,
    scale: float = 0.12,
    band: float = 80.0,
) -> float:
    """Light similar-opponent: mean margin vs prior opponents within Elo band."""
    if not team_hist:
        return 0.0
    close = [h for h in team_hist if abs(float(h["opp_elo"]) - opp_elo) <= band]
    sample = close if len(close) >= 2 else team_hist[-4:]
    if not sample:
        return 0.0
    avg = sum(float(h["margin"]) for h in sample) / len(sample)
    return max(-4.0, min(4.0, avg * scale))


def score_game(
    *,
    elo_pred_margin: float,
    form_home_diff: float,
    form_away_diff: float,
    home_rest: float | None,
    away_rest: float | None,
    conference_game: bool,
    neutral_site: bool,
    rivalry_note: bool,
    similar_home: float,
    similar_away: float,
    weights: dict | None = None,
) -> dict[str, Any]:
    """
    Blend Elo + form + rest/conf/similar into predicted home margin.

    HFA lives primarily inside elo_pred_margin (via Elo hfa_elo). The home_field
    factor is documented for UI; a small absolute nudge remains for non-neutral.
    """
    weights = weights or load_weights()
    factors = {f["key"]: f["weight"] for f in factor_list(weights)}
    elo_w = float(factors.get("elo", 0.48))
    form_w = float(factors.get("recent_form", 0.22))
    hfa_w = float(factors.get("home_field", 0.14))
    rest_w = float(factors.get("rest", 0.06))
    conf_w = float(factors.get("conference_rivalry", 0.05))
    sim_w = float(factors.get("similar_opponent", 0.05))

    elo_cfg = weights.get("elo") or {}
    elo_blend = float(elo_cfg.get("blend", 0.70))

    form_scale = float(weights.get("form_scale", 0.28))
    form_margin = (form_home_diff - form_away_diff) * form_scale

    rest_pts = 0.0
    rest_note = "neutral"
    if home_rest is not None and away_rest is not None:
        gap = home_rest - away_rest
        pts_per = float(weights.get("rest_pts_per_day", 0.15))
        clamp = float(weights.get("rest_clamp", 2.0))
        rest_pts = max(-clamp, min(clamp, gap * pts_per))
        short = float(weights.get("short_week_days", 6))
        short_pts = float(weights.get("short_week_pts", 0.7))
        if home_rest < short and away_rest >= short:
            rest_pts -= short_pts
            rest_note = f"home short week ({home_rest:.0f}d)"
        elif away_rest < short and home_rest >= short:
            rest_pts += short_pts
            rest_note = f"away short week ({away_rest:.0f}d)"
        else:
            rest_note = f"rest gap {gap:+.0f}d"

    conf_adj = -0.2 if conference_game else 0.0
    if rivalry_note:
        bump = float(weights.get("rivalry_margin_bump", 0.35))
        conf_adj += bump if elo_pred_margin >= 0 else -bump

    sim_margin = similar_home - similar_away
    # Small absolute HFA nudge beyond Elo HFA (0 on neutral)
    hfa_nudge = 0.0 if neutral_site else float(weights.get("hfa_extra_points", 0.4))

    margin = (
        elo_blend * elo_pred_margin
        + (1.0 - elo_blend) * form_margin
        + hfa_nudge
        + rest_pts * min(1.0, rest_w / 0.06)
        + conf_adj * min(1.0, conf_w / 0.05)
        + sim_margin * min(1.0, sim_w / 0.05)
    )

    league_total = float(weights.get("league_total", 52.0))
    home_pf = 26.0 + form_home_diff * 0.10 + margin / 2
    away_pf = 26.0 + form_away_diff * 0.10 - margin / 2
    total = max(35.0, min(80.0, 0.40 * (home_pf + away_pf) + 0.60 * league_total))

    breakdown = [
        {
            "factor": "elo",
            "weight": elo_w,
            "effect": round(elo_pred_margin, 2),
            "detail": f"Elo margin {elo_pred_margin:+.1f}",
        },
        {
            "factor": "recent_form",
            "weight": form_w,
            "effect": round(form_margin, 2),
            "detail": f"Form diff home/away {form_home_diff:+.1f}/{form_away_diff:+.1f}",
        },
        {
            "factor": "home_field",
            "weight": hfa_w,
            "effect": round(hfa_nudge, 2),
            "detail": "neutral" if neutral_site else f"Elo HFA + nudge {hfa_nudge:.1f}",
        },
        {
            "factor": "rest",
            "weight": rest_w,
            "effect": round(rest_pts, 2),
            "detail": rest_note,
        },
        {
            "factor": "conference_rivalry",
            "weight": conf_w,
            "effect": round(conf_adj, 2),
            "detail": (
                "conference"
                if conference_game
                else ("rivalry note" if rivalry_note else "non-conf")
            ),
        },
        {
            "factor": "similar_opponent",
            "weight": sim_w,
            "effect": round(sim_margin, 2),
            "detail": f"sim home/away {similar_home:+.1f}/{similar_away:+.1f}",
        },
        {
            "factor": "injuries",
            "weight": float(factors.get("injuries", 0.0)),
            "effect": 0.0,
            "detail": "No reliable CFB injury feed; omitted",
        },
    ]

    return {
        "pred_margin": margin,
        "pred_total": total,
        "pred_home_points": (total + margin) / 2,
        "pred_away_points": (total - margin) / 2,
        "breakdown": breakdown,
    }
