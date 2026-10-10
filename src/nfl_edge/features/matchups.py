"""Matchup engine: combine ratings, HFA, weather, injuries, form, situational."""

from __future__ import annotations

import polars as pl

from nfl_edge.features.ratings import (
    attach_pass_rush_splits,
    build_team_game_log,
    league_avg_points,
    rolling_team_ratings,
)
from nfl_edge.features.elo import build_elo_table
from nfl_edge.features.weather import enrich_schedule_weather, form_weight_for_week, load_weights


def _rating_before_game(rated: pl.DataFrame, team: str, season: int, week: int) -> dict:
    """Team rating going into (season, week): includes every game played before it."""
    prior = rated.filter(
        (pl.col("team") == team)
        & (
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
    )
    if prior.is_empty():
        same = rated.filter(
            (pl.col("team") == team) & (pl.col("season") == season) & (pl.col("week") == week)
        )
        if same.is_empty():
            return {}
        return same.tail(1).to_dicts()[0]
    last = prior.sort(["season", "week"]).tail(1).to_dicts()[0]
    # The last row's plain ratings stop *before* that game; its post_* ratings include it.
    return {**last, **{k.removeprefix("post_"): v for k, v in last.items() if k.startswith("post_")}}


def build_matchup_frame(
    schedules: pl.DataFrame,
    team_stats: pl.DataFrame | None = None,
    injury_features: pl.DataFrame | None = None,
    weights: dict | None = None,
    deep_features: pl.DataFrame | None = None,
    situational_features: pl.DataFrame | None = None,
    *,
    use_injuries: bool = True,
    use_deep: bool = True,
    use_situational: bool = True,
    use_weather: bool = True,
    use_form: bool = True,
) -> pl.DataFrame:
    """
    One row per scheduled game with predicted components.
    Blends points-model with Elo (~55% Elo / 45% points) for stability.

    Feature flags support ablation audits (baseline Elo+points vs layered factors).
    """
    weights = weights or load_weights()
    deep_cfg = weights.get("deep_matchup") or {}
    deep_w = float(deep_cfg.get("weight", 0.35)) if use_deep else 0.0
    sim_scale = float(deep_cfg.get("sim_scale", 0.55))
    mm_scale = float(deep_cfg.get("mismatch_scale", 0.45))
    sit_w = float((weights.get("situational") or {}).get("weight", 1.0)) if use_situational else 0.0
    elo_w = float((weights.get("elo") or {}).get("blend", 0.55))
    games = enrich_schedule_weather(schedules, weights)

    log = build_team_game_log(schedules)
    if team_stats is not None and not team_stats.is_empty():
        log = attach_pass_rush_splits(log, team_stats)
    rated = rolling_team_ratings(log)
    league = league_avg_points(log)
    elo = build_elo_table(schedules)
    elo_by_id = {r["game_id"]: r for r in elo.iter_rows(named=True)} if not elo.is_empty() else {}

    if use_injuries and injury_features is not None and not injury_features.is_empty():
        games = games.join(injury_features, on="game_id", how="left")
        if {"home_inj_points", "away_inj_points"}.issubset(games.columns):
            # Injury points are always <= 0 and every roster carries some injuries, which the
            # ratings already price in. Center on the week's league norm so only unusually
            # banged-up teams move the projection (otherwise every total is shaded Under).
            norm = ((pl.col("home_inj_points") + pl.col("away_inj_points")) / 2.0).mean().over(["season", "week"])
            games = games.with_columns(
                (pl.col("home_inj_points") - norm).alias("home_inj_points"),
                (pl.col("away_inj_points") - norm).alias("away_inj_points"),
            )
    if use_deep and deep_features is not None and not deep_features.is_empty():
        games = games.join(deep_features, on="game_id", how="left")
    if use_situational and situational_features is not None and not situational_features.is_empty():
        games = games.join(situational_features, on="game_id", how="left")

    fill_defaults = {
        "wx_total_adj": 0.0,
        "wx_pass_pct": 0.0,
        "wx_rush_pct": 0.0,
        "hfa_points": 1.4,
        "home_inj_points": 0.0,
        "away_inj_points": 0.0,
        "home_inj_pass_pct": 0.0,
        "away_inj_pass_pct": 0.0,
        "home_inj_rush_pct": 0.0,
        "away_inj_rush_pct": 0.0,
        "home_sim_margin_adj": 0.0,
        "away_sim_margin_adj": 0.0,
        "home_mm_points": 0.0,
        "away_mm_points": 0.0,
        "home_mm_pass_pct": 0.0,
        "away_mm_pass_pct": 0.0,
        "home_mm_rush_pct": 0.0,
        "away_mm_rush_pct": 0.0,
        "deep_net_margin_adj": 0.0,
        "home_sim_continuity": 0.0,
        "away_sim_continuity": 0.0,
        "rest_margin_adj": 0.0,
        "div_margin_adj": 0.0,
        "home_pace_points": 0.0,
        "away_pace_points": 0.0,
        "home_pace_pass_pct": 0.0,
        "away_pace_pass_pct": 0.0,
        "home_pace_rush_pct": 0.0,
        "away_pace_rush_pct": 0.0,
        "pace_total_adj": 0.0,
        "situational_net_margin": 0.0,
    }
    for col, val in fill_defaults.items():
        if col in games.columns:
            games = games.with_columns(pl.col(col).fill_null(val))
        else:
            games = games.with_columns(pl.lit(val).alias(col))

    pred_rows = []
    rating_rows = []
    for r in games.iter_rows(named=True):
        season = int(r["season"])
        week = int(r["week"])
        home_r = _rating_before_game(rated, str(r["home_team"]), season, week)
        away_r = _rating_before_game(rated, str(r["away_team"]), season, week)

        home_off = float(home_r.get("off_rating") or league)
        home_def = float(home_r.get("def_rating") or league)
        away_off = float(away_r.get("off_rating") or league)
        away_def = float(away_r.get("def_rating") or league)
        home_form = float(home_r.get("form_ema") or 0.0)
        away_form = float(away_r.get("form_ema") or 0.0)

        fw = form_weight_for_week(week, weights) if use_form else 0.0
        # Points model: what this offense scores, blended with what that defense allows
        # (def_rating = points allowed, so a leaky defense raises the opponent's projection)
        home_pts = (home_off + away_def) / 2.0
        away_pts = (away_off + home_def) / 2.0
        home_pts += fw * home_form
        away_pts += fw * away_form

        hfa = float(r.get("hfa_points") or 1.4)
        home_pts += hfa / 2.0
        away_pts -= hfa / 2.0

        if use_injuries:
            home_pts += float(r.get("home_inj_points") or 0.0)
            away_pts += float(r.get("away_inj_points") or 0.0)

        wx = float(r.get("wx_total_adj") or 0.0) if use_weather else 0.0
        home_pts += wx / 2.0
        away_pts += wx / 2.0

        # Situational: rest / short week / division / pace (clamped upstream)
        if use_situational:
            sit_margin = sit_w * float(r.get("situational_net_margin") or 0.0)
            sit_margin = max(-4.0, min(4.0, sit_margin))
            home_pts += sit_margin / 2.0
            away_pts -= sit_margin / 2.0
            pace_tot = sit_w * float(r.get("pace_total_adj") or 0.0)
            home_pts += pace_tot / 2.0
            away_pts += pace_tot / 2.0

        # Deep matchup (similar-opponent + player mismatches), weight-clamped
        if use_deep and deep_w > 0:
            home_pts += deep_w * float(r.get("home_sim_margin_adj") or 0.0) * sim_scale
            away_pts += deep_w * float(r.get("away_sim_margin_adj") or 0.0) * sim_scale
            home_pts += deep_w * float(r.get("home_mm_points") or 0.0) * mm_scale
            away_pts += deep_w * float(r.get("away_mm_points") or 0.0) * mm_scale

        home_pts = max(home_pts, 7.0)
        away_pts = max(away_pts, 7.0)
        points_margin = home_pts - away_pts
        points_total = home_pts + away_pts

        # Elo blend
        er = elo_by_id.get(r["game_id"], {})
        elo_margin = float(er.get("elo_pred_margin") or points_margin)
        margin = elo_w * elo_margin + (1.0 - elo_w) * points_margin
        # keep total from points model (Elo doesn't predict totals well alone)
        total = points_total
        # rescale team points to match blended margin
        home_pts = (total + margin) / 2.0
        away_pts = (total - margin) / 2.0

        base_pass = 230.0
        base_rush = 115.0
        wx_pass = float(r.get("wx_pass_pct") or 0.0) if use_weather else 0.0
        wx_rush = float(r.get("wx_rush_pct") or 0.0) if use_weather else 0.0
        inj_h_pass = float(r.get("home_inj_pass_pct") or 0.0) if use_injuries else 0.0
        inj_a_pass = float(r.get("away_inj_pass_pct") or 0.0) if use_injuries else 0.0
        inj_h_rush = float(r.get("home_inj_rush_pct") or 0.0) if use_injuries else 0.0
        inj_a_rush = float(r.get("away_inj_rush_pct") or 0.0) if use_injuries else 0.0
        pace_h_pass = sit_w * float(r.get("home_pace_pass_pct") or 0.0) if use_situational else 0.0
        pace_a_pass = sit_w * float(r.get("away_pace_pass_pct") or 0.0) if use_situational else 0.0
        pace_h_rush = sit_w * float(r.get("home_pace_rush_pct") or 0.0) if use_situational else 0.0
        pace_a_rush = sit_w * float(r.get("away_pace_rush_pct") or 0.0) if use_situational else 0.0
        home_pass = base_pass * (
            1
            + wx_pass
            + inj_h_pass
            + pace_h_pass
            + deep_w * float(r.get("home_mm_pass_pct") or 0.0) * mm_scale
        )
        away_pass = base_pass * (
            1
            + wx_pass
            + inj_a_pass
            + pace_a_pass
            + deep_w * float(r.get("away_mm_pass_pct") or 0.0) * mm_scale
        )
        home_rush = base_rush * (
            1
            + wx_rush
            + inj_h_rush
            + pace_h_rush
            + deep_w * float(r.get("home_mm_rush_pct") or 0.0) * mm_scale
        )
        away_rush = base_rush * (
            1
            + wx_rush
            + inj_a_rush
            + pace_a_rush
            + deep_w * float(r.get("away_mm_rush_pct") or 0.0) * mm_scale
        )

        home_pass *= home_pts / league
        away_pass *= away_pts / league
        home_rush *= 0.7 + 0.3 * home_pts / league
        away_rush *= 0.7 + 0.3 * away_pts / league

        if home_r.get("roll_pass_yards"):
            home_pass = 0.5 * home_pass + 0.5 * float(home_r["roll_pass_yards"])
        if away_r.get("roll_pass_yards"):
            away_pass = 0.5 * away_pass + 0.5 * float(away_r["roll_pass_yards"])
        if home_r.get("roll_rush_yards"):
            home_rush = 0.5 * home_rush + 0.5 * float(home_r["roll_rush_yards"])
        if away_r.get("roll_rush_yards"):
            away_rush = 0.5 * away_rush + 0.5 * float(away_r["roll_rush_yards"])

        rating_rows.append(
            {
                "game_id": r["game_id"],
                "home_off": home_off,
                "home_def": home_def,
                "away_off": away_off,
                "away_def": away_def,
                "home_form": home_form,
                "away_form": away_form,
                "elo_home": er.get("elo_home"),
                "elo_away": er.get("elo_away"),
            }
        )
        pred_rows.append(
            {
                "game_id": r["game_id"],
                "pred_home_points": round(home_pts, 2),
                "pred_away_points": round(away_pts, 2),
                "pred_margin": round(margin, 2),
                "pred_total": round(total, 2),
                "pred_home_pass_yds": round(home_pass, 1),
                "pred_away_pass_yds": round(away_pass, 1),
                "pred_home_rush_yds": round(home_rush, 1),
                "pred_away_rush_yds": round(away_rush, 1),
                "form_weight": fw,
            }
        )

    return (
        games.join(pl.DataFrame(rating_rows), on="game_id", how="left").join(
            pl.DataFrame(pred_rows), on="game_id", how="left"
        )
    )
