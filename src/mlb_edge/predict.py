"""MLB prediction + deep weighted matchups."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import polars as pl

from mlb_edge.config import OUTPUT_DIR, current_mlb_season, ensure_dirs
from mlb_edge.ingest import (
    completed_games,
    fetch_pitcher_stats,
    fetch_team_season_stats,
    upcoming_and_recent,
)
from mlb_edge.weights import factor_list, load_weights, save_weights
from mlb_edge.scoring import predict_runs
from mlb_edge.elo import build_mlb_elo, elo_run_expectation
from nfl_edge.odds_math import expected_value, implied_to_american, margin_to_home_win_prob, prob_over_line


def _park_factor(team_abbr: str | None, weights: dict) -> float:
    parks = weights.get("park_run_factors") or {}
    if not team_abbr:
        return 1.0
    return float(parks.get(str(team_abbr).upper(), 100)) / 100.0


def _sp_run_suppression(sp: dict) -> float:
    """
    Map SP quality → expected runs allowed contribution (lower ERA → fewer runs).
    Returns adjustment vs league ~4.5 ERA in runs per 9 scaled to start (~5.5 IP).
    """
    era = float(sp.get("era") or 4.5)
    whip = float(sp.get("whip") or 1.35)
    k9 = float(sp.get("k_per_9") or 8.0)
    bb9 = float(sp.get("bb_per_9") or 3.2)
    # quality score: negative is better pitcher
    q = (era - 4.20) * 0.55 + (whip - 1.30) * 1.2 - (k9 - 8.5) * 0.08 + (bb9 - 3.0) * 0.12
    # convert to runs allowed in a start (~5.5 IP)
    return q * (5.5 / 9.0)


def _weather_runs(temp: float | None, wind: str | None) -> float:
    adj = 0.0
    if temp is not None:
        if temp >= 85:
            adj += 0.25
        elif temp <= 50:
            adj -= 0.30
    if wind:
        w = wind.lower()
        if "out" in w:
            adj += 0.35
        elif "in" in w:
            adj -= 0.30
    return adj


def _team_offense_runs(team_row: dict | None, league_rpg: float = 4.5) -> float:
    if not team_row:
        return league_rpg
    rpg = float(team_row.get("rpg") or league_rpg)
    ops = float(team_row.get("ops") or 0.720)
    # blend RPG with OPS vs league
    return 0.7 * rpg + 0.3 * (league_rpg + (ops - 0.720) * 8.0)


def _bullpen_runs(team_row: dict | None) -> float:
    if not team_row:
        return 0.0
    era = float(team_row.get("era") or 4.2)
    # bullpen proxy from team pitching ERA — imperfect but directional
    return (era - 4.0) * 0.15


def predict_mlb_slate(days_ahead: int = 2, days_back: int = 60) -> dict[str, Any]:
    ensure_dirs()
    weights = load_weights()
    save_weights(weights)  # ensure file exists for UI
    factors = {f["key"]: f["weight"] for f in factor_list(weights)}

    sched = upcoming_and_recent(days_ahead=days_ahead, days_back=days_back)
    if sched.is_empty():
        return {"sport": "MLB", "games": [], "edges": [], "factors": factor_list(weights), "weights": weights}

    season = current_mlb_season()
    team_stats = fetch_team_season_stats(season)
    team_by_id = {
        int(r["team_id"]): r for r in team_stats.iter_rows(named=True)
    } if not team_stats.is_empty() else {}
    # also map abbreviation via schedule rows we already have
    abbr_by_id: dict[int, str] = {}
    for r in sched.iter_rows(named=True):
        if r.get("home_id"):
            abbr_by_id[int(r["home_id"])] = str(r["home_team"]).upper()
        if r.get("away_id"):
            abbr_by_id[int(r["away_id"])] = str(r["away_team"]).upper()
    for tid, row in team_by_id.items():
        if tid in abbr_by_id:
            row["team"] = abbr_by_id[tid]
    team_by_abbr = {
        str(r["team"]).upper(): r for r in team_by_id.values()
    }

    today = date.today().isoformat()
    upcoming = sched.filter(
        (pl.col("gameday") >= today)
        & pl.col("status").is_in(
            ["Scheduled", "Pre-Game", "Warmup", "In Progress", "Preview", "Delayed Start"]
        )
        | (
            (pl.col("gameday") >= today)
            & pl.col("home_score").is_null()
        )
    )
    # broader: any game today+ without final scores
    upcoming = sched.filter((pl.col("gameday") >= today) & pl.col("home_score").is_null())
    if upcoming.is_empty():
        # show tomorrow slate even if scored weirdly — take next 2 days of games
        upcoming = sched.filter(pl.col("gameday") >= today).head(20)

    completed = completed_games(sched)
    form = _rolling_form(completed, window=15)
    elo_k = float(weights.get("elo_k", 6.0))
    elo_ratings = build_mlb_elo(completed, k=elo_k) if not completed.is_empty() else {}
    form_w = float(weights.get("form_blend", 0.45))
    elo_w = float(weights.get("elo_blend", 0.45))
    league_w = max(0.0, 1.0 - form_w - elo_w)
    conf = weights.get("confidence") or {}
    min_margin = float(conf.get("min_abs_margin", 0.5))
    min_sp_gap = float(conf.get("min_sp_era_gap", 1.0))

    games_out: list[dict] = []
    pitcher_cache: dict[int, dict] = {}

    for r in upcoming.iter_rows(named=True):
        home = str(r.get("home_team") or "").upper()
        away = str(r.get("away_team") or "").upper()
        home_row = team_by_id.get(int(r["home_id"])) if r.get("home_id") else team_by_abbr.get(home)
        away_row = team_by_id.get(int(r["away_id"])) if r.get("away_id") else team_by_abbr.get(away)

        home_sp_id = r.get("home_sp_id")
        away_sp_id = r.get("away_sp_id")
        home_sp = {}
        away_sp = {}
        if home_sp_id:
            home_sp = pitcher_cache.setdefault(int(home_sp_id), fetch_pitcher_stats(int(home_sp_id), season))
        if away_sp_id:
            away_sp = pitcher_cache.setdefault(int(away_sp_id), fetch_pitcher_stats(int(away_sp_id), season))

        # Walk-forward blend: recent form + Elo (avoid season-stat leakage)
        home_form = float(form.get(home, {}).get("rpg", 4.5) or 4.5)
        away_form = float(form.get(away, {}).get("rpg", 4.5) or 4.5)
        eh = elo_ratings.get(home, 1500.0)
        ea = elo_ratings.get(away, 1500.0)
        home_rpg = form_w * home_form + elo_w * elo_run_expectation(eh, ea) + league_w * 4.5
        away_rpg = form_w * away_form + elo_w * elo_run_expectation(ea, eh) + league_w * 4.5

        park = _park_factor(home, weights)
        home_bp_era = float((home_row or {}).get("era") or 4.2)
        away_bp_era = float((away_row or {}).get("era") or 4.2)

        scored = predict_runs(
            home_rpg=home_rpg,
            away_rpg=away_rpg,
            home_sp=home_sp,
            away_sp=away_sp,
            park_factor=park,
            temp=r.get("temp"),
            wind=r.get("wind"),
            home_bp_era=home_bp_era,
            away_bp_era=away_bp_era,
            weights=weights,
        )
        home_runs = scored["pred_home_runs"]
        away_runs = scored["pred_away_runs"]
        margin = scored["pred_margin"]
        total = scored["pred_total"]
        sigma = float(weights.get("run_sigma", 3.2))
        home_win = margin_to_home_win_prob(margin, sigma)
        breakdown = scored["breakdown"]
        era_gap = abs(float(home_sp.get("era") or 4.5) - float(away_sp.get("era") or 4.5)) if home_sp and away_sp else 0.0
        high_conf = bool(home_sp and away_sp and era_gap >= min_sp_gap and abs(margin) >= min_margin)

        games_out.append(
            {
                "game_pk": r.get("game_pk"),
                "gameday": r.get("gameday"),
                "home_team": home,
                "away_team": away,
                "home_name": r.get("home_name"),
                "away_name": r.get("away_name"),
                "venue": r.get("venue"),
                "home_sp_name": home_sp.get("name") or r.get("home_sp_name"),
                "away_sp_name": away_sp.get("name") or r.get("away_sp_name"),
                "home_sp_era": home_sp.get("era"),
                "away_sp_era": away_sp.get("era"),
                "home_sp_throws": home_sp.get("throws"),
                "away_sp_throws": away_sp.get("throws"),
                "sp_era_gap": round(era_gap, 2),
                "pred_home_runs": round(home_runs, 2),
                "pred_away_runs": round(away_runs, 2),
                "pred_margin": round(margin, 2),
                "pred_total": round(total, 2),
                "home_win_prob": round(home_win, 4),
                "high_confidence": high_conf,
                "temp": r.get("temp"),
                "wind": r.get("wind"),
                "factor_breakdown": breakdown,
                "suggested_bets": _suggest_bets(
                    home, away, margin, total, home_win, sigma, weights, high_conf=high_conf
                ),
            }
        )

    # sort by gameday
    games_out.sort(key=lambda g: (str(g.get("gameday")), str(g.get("away_team"))))

    edges = []
    for g in games_out:
        edges.extend(g.get("suggested_bets") or [])

    out = {
        "sport": "MLB",
        "season": season,
        "games": games_out,
        "edges": edges,
        "factors": factor_list(weights),
        "weights": weights,
    }

    if games_out:
        pl.DataFrame(
            [
                {k: v for k, v in g.items() if k not in {"factor_breakdown", "suggested_bets"}}
                for g in games_out
            ]
        ).write_csv(OUTPUT_DIR / f"mlb_{date.today().isoformat()}_predictions.csv")

    return out


def _platoon_adj(throws: str | None, offense_row: dict | None) -> float:
    """Tiny bump — without full split data, use OPS as stand-in + handedness note."""
    if not throws or not offense_row:
        return 0.0
    ops = float(offense_row.get("ops") or 0.72)
    # Strong OPS teams punish weaker pitching more; handedness nuance deferred
    return (ops - 0.720) * 1.5


def _similar_sp_signal(team_form: dict, sp: dict) -> float:
    """If team has been hitting well, mild bump vs mediocre SP; shrink always."""
    if not sp:
        return 0.0
    era = float(sp.get("era") or 4.5)
    rpg = float(team_form.get("rpg") or 4.5)
    # good offense vs bad SP
    raw = (rpg - 4.5) * 0.15 + (era - 4.2) * 0.12
    return max(-0.4, min(0.4, raw))


def _rolling_form(completed: pl.DataFrame, window: int = 12) -> dict[str, dict]:
    if completed.is_empty():
        return {}
    rows = []
    for r in completed.iter_rows(named=True):
        rows.append({"team": str(r["home_team"]).upper(), "gameday": r["gameday"], "runs": float(r["home_score"])})
        rows.append({"team": str(r["away_team"]).upper(), "gameday": r["gameday"], "runs": float(r["away_score"])})
    df = pl.DataFrame(rows).sort(["team", "gameday"])
    out: dict[str, dict] = {}
    for team in df["team"].unique().to_list():
        sub = df.filter(pl.col("team") == team).tail(window)
        out[team] = {"rpg": float(sub["runs"].mean()), "n": sub.height}
    return out


def _suggest_bets(
    home: str,
    away: str,
    margin: float,
    total: float,
    home_win: float,
    sigma: float,
    weights: dict,
    *,
    high_conf: bool = False,
) -> list[dict]:
    """Model-only suggestions at standard -110 prices (no live book required)."""
    from nfl_edge.odds_math import normal_cdf

    thr = weights.get("edge_thresholds") or {}
    bets: list[dict] = []

    def maybe_add(market: str, label: str, prob: float, min_ev_key: str) -> None:
        prob = min(max(float(prob), 0.01), 0.99)
        price = -110
        ev = expected_value(prob, price)
        # Raise EV bar slightly for low-confidence games; ease it for high-conf
        floor = float(thr.get(min_ev_key, 0.04))
        if high_conf:
            floor *= 0.75
        else:
            floor *= 1.35
        if ev >= floor:
            bets.append(
                {
                    "market": market,
                    "label": label,
                    "model_prob": round(prob, 4),
                    "fair_american": implied_to_american(prob),
                    "dk_price": price,
                    "ev": round(ev, 4),
                    "home_team": home,
                    "away_team": away,
                    "passes_threshold": True,
                    "high_confidence": high_conf,
                }
            )

    maybe_add("h2h", f"{home} ML", home_win, "h2h_min_ev")
    maybe_add("h2h", f"{away} ML", 1 - home_win, "h2h_min_ev")

    line = round(total * 2) / 2
    p_over = prob_over_line(total, float(weights.get("run_sigma", 3.2)), line)
    maybe_add("totals", f"Over {line}", p_over, "totals_min_ev")
    maybe_add("totals", f"Under {line}", 1 - p_over, "totals_min_ev")

    if margin >= 0:
        p_fav = 1.0 - normal_cdf(1.5, margin, sigma)
        maybe_add("spreads", f"{home} -1.5", p_fav, "spreads_min_ev")
        maybe_add("spreads", f"{away} +1.5", 1 - p_fav, "spreads_min_ev")
    else:
        p_fav = normal_cdf(-1.5, margin, sigma)
        maybe_add("spreads", f"{away} -1.5", p_fav, "spreads_min_ev")
        maybe_add("spreads", f"{home} +1.5", 1 - p_fav, "spreads_min_ev")

    return bets
