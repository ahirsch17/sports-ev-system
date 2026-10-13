"""Build an NFL week slate for the UI / API layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import polars as pl

from nfl_edge.config import OUTPUT_DIR, current_nfl_season, ensure_dirs
from nfl_edge.features.deep_matchups import build_deep_matchup_features, build_team_style_table
from nfl_edge.features.injuries import build_injury_features
from nfl_edge.features.ratings import build_team_game_log
from nfl_edge.features.situational import build_situational_features
from nfl_edge.features.weather import load_weights
from nfl_edge.models.props import is_extra_playmaker, listed_receiving
from nfl_edge.ingest.nflverse import (
    data_as_of,
    load_depth_charts,
    load_injuries,
    load_player_stats,
    load_schedules,
    load_team_stats,
)
from nfl_edge.ingest.odds import flatten_main_markets, load_live_events
from nfl_edge.ingest.schedule_odds import schedule_lines_as_odds
from nfl_edge.models.game import predict_slate
from nfl_edge.models.props import project_slate_props
from nfl_edge.edge.calculator import build_parlays, compute_main_edges, normalize_team


def resolve_week(season: int | None = None, week: int | None = None) -> tuple[int, int]:
    season = season or current_nfl_season()
    if week is None:
        sched = upcoming_games(load_schedules([season], regular_only=True))
        week = int(sched["week"].min()) if not sched.is_empty() else int(load_schedules([season])["week"].max())
    return season, week


def _kickoff_utc(gameday: Any, gametime: Any) -> datetime | None:
    """nflverse lists kickoff as an Eastern date + HH:MM."""
    try:
        from zoneinfo import ZoneInfo

        local = datetime.strptime(f"{gameday} {gametime or '13:00'}", "%Y-%m-%d %H:%M")
        return local.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
    except Exception:
        return None


def upcoming_games(sched: pl.DataFrame) -> pl.DataFrame:
    """Games with no final score that haven't started yet (in-progress games can't be bet pregame)."""
    pending = sched.filter(pl.col("home_score").is_null())
    if pending.is_empty() or "gameday" not in pending.columns:
        return pending
    now = datetime.now(timezone.utc)
    keep = []
    for r in pending.iter_rows(named=True):
        kick = _kickoff_utc(r.get("gameday"), r.get("gametime"))
        keep.append(kick is None or kick > now)
    return pending.filter(pl.Series(keep))


def market_lines(week_sched: pl.DataFrame, *, force_odds: bool = False) -> tuple[list[dict], dict[str, Any]]:
    """
    Live DraftKings lines (The Odds API) where available, nflverse schedule lines
    for any game DraftKings hasn't posted — or for everything when there's no key.
    """
    sched_rows = schedule_lines_as_odds(week_sched)
    events, meta = load_live_events(force=force_odds)
    dk_rows = flatten_main_markets(events) if events else []
    covered = {(normalize_team(str(r["home_team"])), normalize_team(str(r["away_team"]))) for r in dk_rows}
    week_pairs = {(str(r["home_team"]), str(r["away_team"])) for r in week_sched.iter_rows(named=True)}
    dk_games = len(covered & week_pairs)
    fallback = [
        r
        for r in sched_rows
        if (normalize_team(str(r["home_team"])), normalize_team(str(r["away_team"]))) not in covered
    ]
    info = {
        "odds_source": "draftkings" if dk_games else "nflverse_lines",
        "dk_games": dk_games,
        "odds_as_of": meta.get("fetched_at") if dk_games else None,
        "odds_requests_remaining": meta.get("requests_remaining"),
        "odds_error": meta.get("error"),
    }
    return dk_rows + fallback, info


def _balanced_game_props(props: list[dict], *, limit: int = 10) -> list[dict]:
    """
    Keep a readable mix: each team's QB pass, top rush, top recv — not just home side.
    """
    if not props:
        return []
    by_team: dict[str, list[dict]] = {}
    for p in props:
        by_team.setdefault(str(p.get("team") or "?"), []).append(p)

    picked: list[dict] = []
    seen: set[tuple] = set()

    def _take(team_props: list[dict], market: str, n: int) -> None:
        rows = [p for p in team_props if str(p.get("market")) == market]
        rows.sort(key=lambda p: float(p.get("mu") or 0), reverse=True)
        for p in rows[:n]:
            key = (p.get("player"), p.get("market"))
            if key in seen:
                continue
            seen.add(key)
            picked.append(p)

    # Pass (1 each), one receiver playmaker plus any target-drop under, rush (1 each)
    for team_props in by_team.values():
        _take(team_props, "player_pass_yds", 1)
    for team_props in by_team.values():
        for p in listed_receiving(team_props):
            key = (p.get("player"), p.get("market"))
            if key in seen:
                continue
            seen.add(key)
            picked.append(p)
    for team_props in by_team.values():
        _take(team_props, "player_rush_yds", 1)

    # Fill remainder by mu if under limit. Do not add a second playmaker.
    rest = sorted(props, key=lambda p: float(p.get("mu") or 0), reverse=True)
    for p in rest:
        if len(picked) >= limit:
            break
        key = (p.get("player"), p.get("market"))
        if key in seen or is_extra_playmaker(p, picked):
            continue
        seen.add(key)
        picked.append(p)
    return picked[:limit]


@dataclass
class WeekModel:
    """Full-model outputs for one NFL week (the audited stack used by the board and bridge)."""

    season: int
    week: int
    weights: dict
    preds: pl.DataFrame
    week_sched: pl.DataFrame
    inj_feat: pl.DataFrame
    deep: pl.DataFrame
    sit: pl.DataFrame
    player_stats: pl.DataFrame
    depth: pl.DataFrame
    injuries: pl.DataFrame | None = None


def model_week(
    season: int,
    week: int,
    *,
    weights: dict | None = None,
    upcoming_only: bool = True,
    schedules_hist: pl.DataFrame | None = None,
) -> WeekModel | None:
    """Run the full NFL model (ratings, form, weather, injuries, deep, situational) for one week."""
    weights = weights or load_weights()
    hist_seasons = list(range(max(2006, season - 3), season + 1))
    if schedules_hist is None:
        schedules_hist = load_schedules(hist_seasons)
    week_sched = schedules_hist.filter((pl.col("season") == season) & (pl.col("week") == week))
    if upcoming_only:
        week_sched = upcoming_games(week_sched)
    if week_sched.is_empty():
        return None

    team_stats = load_team_stats(hist_seasons)
    player_stats = load_player_stats(hist_seasons)
    injuries = load_injuries([season])
    depth = load_depth_charts([season])
    inj_feat = build_injury_features(week_sched, injuries, depth, weights, player_stats=player_stats)
    team_games = build_team_game_log(schedules_hist)
    deep = build_deep_matchup_features(
        week_sched, team_stats, player_stats, depth, team_games, weights=weights
    )
    styles = build_team_style_table(team_stats) if not team_stats.is_empty() else None
    sit = build_situational_features(
        week_sched, styles=styles, weights=weights, history=schedules_hist
    )

    preds_all = predict_slate(
        schedules_hist,
        team_stats,
        inj_feat,
        weights,
        deep_features=deep,
        situational_features=sit,
        player_stats=player_stats,
    )
    week_ids = week_sched["game_id"].to_list()
    preds = preds_all.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("game_id").is_in(week_ids)
    )
    return WeekModel(
        season=season,
        week=week,
        weights=weights,
        preds=preds,
        week_sched=week_sched,
        inj_feat=inj_feat,
        deep=deep,
        sit=sit,
        player_stats=player_stats,
        depth=depth,
        injuries=injuries,
    )


def build_nfl_slate(
    season: int | None = None, week: int | None = None, *, force_odds: bool = False
) -> dict[str, Any]:
    """Return predictions, deep matchups, props, edges, parlays for a week."""
    ensure_dirs()
    season, week = resolve_week(season, week)
    weights = load_weights()
    hist_seasons = list(range(max(2006, season - 3), season + 1))
    schedules_hist = load_schedules(hist_seasons)
    data_info: dict[str, Any] = {"nflverse_as_of": data_as_of(season)}

    # Betting board = games that haven't kicked off (finished or in-progress games stay off)
    wm = model_week(season, week, weights=weights, schedules_hist=schedules_hist)
    if wm is None:
        # whole week done - advance to next unfinished week
        season, week = resolve_week(season, None)
        wm = model_week(season, week, weights=weights)
        if wm is None:
            return {
                "season": season, "week": week, "games": [], "edges": [], "parlays": [], "props": [],
                "data": data_info,
            }

    week_sched, preds = wm.week_sched, wm.preds
    inj_feat, deep, sit = wm.inj_feat, wm.deep, wm.sit
    props = project_slate_props(preds, wm.player_stats, depth_charts=wm.depth, injuries=wm.injuries)

    odds_rows, odds_info = market_lines(week_sched, force_odds=force_odds)
    data_info.update(odds_info)
    edges = compute_main_edges(preds, odds_rows, weights)
    parlays = build_parlays(edges) if not edges.is_empty() else []

    # persist for CLI compatibility
    preds.write_csv(OUTPUT_DIR / f"week_{season}_{week}_predictions.csv")
    if not props.is_empty():
        props.write_csv(OUTPUT_DIR / f"week_{season}_{week}_props.csv")
    if not deep.is_empty():
        deep.write_csv(OUTPUT_DIR / f"week_{season}_{week}_deep_matchups.csv")
    if not edges.is_empty():
        edges.write_csv(OUTPUT_DIR / f"week_{season}_{week}_edges.csv")

    games = []
    deep_by_id = {r["game_id"]: r for r in deep.iter_rows(named=True)} if not deep.is_empty() else {}
    inj_by_id = {r["game_id"]: r for r in inj_feat.iter_rows(named=True)} if not inj_feat.is_empty() else {}
    sit_by_id = {r["game_id"]: r for r in sit.iter_rows(named=True)} if not sit.is_empty() else {}
    conf_min = float((weights.get("confidence") or {}).get("min_abs_margin", 5.0))
    props_by_game: dict[str, list] = {}
    if not props.is_empty():
        for p in props.iter_rows(named=True):
            props_by_game.setdefault(str(p["game_id"]), []).append(p)

    for r in preds.iter_rows(named=True):
        d = deep_by_id.get(r["game_id"], {})
        inj = inj_by_id.get(r["game_id"], {})
        srow = sit_by_id.get(r["game_id"], {})
        games.append(
            {
                **{k: r.get(k) for k in r},
                "high_confidence": abs(float(r.get("pred_margin") or 0)) >= conf_min,
                "home_qb_out": inj.get("home_qb_out"),
                "away_qb_out": inj.get("away_qb_out"),
                "home_inj_points": inj.get("home_inj_points"),
                "away_inj_points": inj.get("away_inj_points"),
                "home_inj_note": inj.get("home_inj_note"),
                "away_inj_note": inj.get("away_inj_note"),
                "deep_net_margin_adj": d.get("deep_net_margin_adj"),
                "home_sim_continuity": d.get("home_sim_continuity"),
                "away_sim_continuity": d.get("away_sim_continuity"),
                "home_sim_note": d.get("home_sim_note"),
                "away_sim_note": d.get("away_sim_note"),
                "home_mm_note": d.get("home_mm_note"),
                "away_mm_note": d.get("away_mm_note"),
                "situational_net_margin": srow.get("situational_net_margin"),
                "rest_note": srow.get("rest_note"),
                "div_note": srow.get("div_note"),
                "near_miss_note": srow.get("near_miss_note"),
                "near_miss_margin_adj": srow.get("near_miss_margin_adj"),
                "home_pace_note": srow.get("home_pace_note"),
                "away_pace_note": srow.get("away_pace_note"),
                "player_props": _balanced_game_props(props_by_game.get(str(r["game_id"]), []), limit=10),
                "factor_breakdown": _nfl_factor_breakdown(r, d, weights, inj, srow),
            }
        )

    return {
        "sport": "NFL",
        "season": season,
        "week": week,
        "games": games,
        "props": props.to_dicts() if not props.is_empty() else [],
        "edges": edges.to_dicts() if not edges.is_empty() else [],
        "parlays": parlays,
        "weights": weights,
        "data": data_info,
    }


def _nfl_factor_breakdown(
    pred: dict,
    deep: dict,
    weights: dict,
    inj: dict | None = None,
    sit: dict | None = None,
) -> list[dict]:
    """Human-readable weighted factors for the UI."""
    inj = inj or {}
    sit = sit or {}
    hfa = float(weights.get("hfa_points", {}).get("2021_plus", 1.4))
    deep_w = float((weights.get("deep_matchup") or {}).get("weight", 1.0))
    sit_w = float((weights.get("situational") or {}).get("weight", 1.0))
    home_note = inj.get("home_inj_note") or pred.get("home_inj_note") or "n/a"
    away_note = inj.get("away_inj_note") or pred.get("away_inj_note") or "n/a"
    home_pts = inj.get("home_inj_points")
    away_pts = inj.get("away_inj_points")
    if home_pts is None:
        home_pts = pred.get("home_inj_points")
    if away_pts is None:
        away_pts = pred.get("away_inj_points")
    inj_effect = f"{home_note} | {away_note}"
    if home_pts is not None or away_pts is not None:
        inj_effect += (
            f" (pts home={float(home_pts or 0):+.1f}, away={float(away_pts or 0):+.1f})"
        )
    qb_bits = []
    if float(inj.get("home_qb_out") or pred.get("home_qb_out") or 0) >= 0.35:
        qb_bits.append("home QB flagged")
    if float(inj.get("away_qb_out") or pred.get("away_qb_out") or 0) >= 0.35:
        qb_bits.append("away QB flagged")
    if qb_bits:
        inj_effect += f"; {', '.join(qb_bits)}"
    plan = " ".join(
        str(note)
        for note in (pred.get("home_script_note"), pred.get("away_script_note"))
        if note
    )
    if plan:
        inj_effect += f". Plan: {plan}"
    sit_effect = (
        f"{sit.get('rest_note') or 'rest n/a'}; {sit.get('div_note') or 'div n/a'}; "
        f"pace {sit.get('home_pace_note') or 'n/a'} vs {sit.get('away_pace_note') or 'n/a'} "
        f"(net={float(sit.get('situational_net_margin') or 0):+.1f})"
    )
    return [
        {"factor": "Team ratings (off/def)", "weight": 0.35, "effect": "Base margin from rolling points for/against"},
        {"factor": "Home field", "weight": 0.12, "effect": f"~{hfa:+.1f} pts to home (era-weighted)"},
        {"factor": "Recent form", "weight": 0.10, "effect": "Last-3 margin EMA, phase-weighted"},
        {"factor": "Weather", "weight": 0.10, "effect": f"Wind/temp/roof -> total & pass (wind={pred.get('wind')})"},
        {"factor": "Injuries / depth", "weight": 0.12, "effect": inj_effect},
        {
            "factor": "Rest / division / pace",
            "weight": 0.08 * sit_w,
            "effect": sit_effect,
        },
        {
            "factor": "Similar-opponent history",
            "weight": 0.12 * deep_w,
            "effect": (
                f"deep={deep.get('deep_net_margin_adj', 0):+.1f}; "
                f"continuity home/away "
                f"{float(deep.get('home_sim_continuity') or 0):.0%}/"
                f"{float(deep.get('away_sim_continuity') or 0):.0%}"
            ),
        },
        {
            "factor": "Player mismatches",
            "weight": 0.09 * deep_w,
            "effect": f"{deep.get('home_mm_note') or 'n/a'} vs {deep.get('away_mm_note') or 'n/a'}",
        },
    ]
