"""CFB slate prediction + walk-forward scoring (no leakage)."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from cfb_edge.config import OUTPUT_DIR, current_cfb_season, ensure_dirs
from cfb_edge.elo import expected_score
from cfb_edge.ingest import data_source_used, load_schedules
from cfb_edge.scoring import score_game, similar_opp_signal
from cfb_edge.weights import factor_list, load_weights, save_weights
from nfl_edge.odds_math import expected_value, implied_to_american, margin_to_home_win_prob, normal_cdf, prob_over_line


def _rivalry_flag(notes: Any, home: str, away: str) -> bool:
    if not notes:
        return False
    s = str(notes).lower()
    keys = ("rivalry", "trophy", "classic", "iron bowl", "red river", "egg bowl", "civil war")
    return any(k in s for k in keys)


def predict_games_walkforward(
    schedules: pl.DataFrame,
    weights: dict | None = None,
    *,
    both_fbs_eval: bool = True,
) -> pl.DataFrame:
    """
    Chronological walk-forward predictions for every game in schedules.
    Elo/form/similar history only use prior completed games.
    """
    weights = weights or load_weights()
    elo_cfg = weights.get("elo") or {}
    k = float(elo_cfg.get("k", 32.0))
    hfa_elo = float(elo_cfg.get("hfa_elo", 72.0))
    ppe = float(elo_cfg.get("points_per_elo", 0.04))
    cupcake_k = float(elo_cfg.get("cupcake_k_scale", 0.35))
    season_regress = float(elo_cfg.get("season_regress", 0.25))
    form_n = int(weights.get("form_games", 4))
    sim_scale = float(weights.get("similar_scale", 0.12))
    sigma = float(weights.get("margin_sigma", 16.5))

    ratings: dict[str, float] = {}
    form_hist: dict[str, list[tuple[float, float]]] = {}
    sim_hist: dict[str, list[dict]] = {}
    last_day: dict[str, date] = {}
    last_season: int | None = None
    rows: list[dict[str, Any]] = []

    def get(t: str) -> float:
        return ratings.setdefault(t, 1500.0)

    games = schedules.sort(["season", "week", "gameday", "game_id"])
    for r in games.iter_rows(named=True):
        season = int(r["season"])
        if last_season is not None and season != last_season and season_regress > 0:
            for t, val in list(ratings.items()):
                ratings[t] = val + season_regress * (1500.0 - val)
        last_season = season

        home = str(r["home_team"])
        away = str(r["away_team"])
        rh, ra = get(home), get(away)
        neutral = bool(r.get("neutral_site"))
        hfa = 0.0 if neutral else hfa_elo
        elo_margin = (rh + hfa - ra) * ppe

        # rolling form from prior games only
        def form_diff(team: str) -> float:
            hist = form_hist.get(team) or []
            if not hist:
                return 0.0
            tail = hist[-form_n:]
            pf = sum(x[0] for x in tail) / len(tail)
            pa = sum(x[1] for x in tail) / len(tail)
            return pf - pa

        fh, fa = form_diff(home), form_diff(away)

        # rest
        day_s = str(r.get("gameday") or "")[:10]
        try:
            day = date.fromisoformat(day_s) if day_s else None
        except ValueError:
            day = None
        home_rest = (day - last_day[home]).days if day and home in last_day else None
        away_rest = (day - last_day[away]).days if day and away in last_day else None

        conf_game = bool(r.get("conference_game"))
        rivalry = _rivalry_flag(r.get("notes"), home, away)
        sim_h = similar_opp_signal(ra, sim_hist.get(home) or [], scale=sim_scale)
        sim_a = similar_opp_signal(rh, sim_hist.get(away) or [], scale=sim_scale)

        scored = score_game(
            elo_pred_margin=elo_margin,
            form_home_diff=fh,
            form_away_diff=fa,
            home_rest=float(home_rest) if home_rest is not None else None,
            away_rest=float(away_rest) if away_rest is not None else None,
            conference_game=conf_game,
            neutral_site=neutral,
            rivalry_note=rivalry,
            similar_home=sim_h,
            similar_away=sim_a,
            weights=weights,
        )
        margin = float(scored["pred_margin"])
        total = float(scored["pred_total"])
        home_win = margin_to_home_win_prob(margin, sigma)

        both_fbs = bool(r.get("both_fbs", True))
        rows.append(
            {
                "game_id": r.get("game_id"),
                "season": season,
                "week": r.get("week"),
                "gameday": r.get("gameday"),
                "home_team": home,
                "away_team": away,
                "home_conference": r.get("home_conference"),
                "away_conference": r.get("away_conference"),
                "neutral_site": neutral,
                "conference_game": conf_game,
                "is_cupcake": bool(r.get("is_cupcake")),
                "both_fbs": both_fbs,
                "elo_home": rh,
                "elo_away": ra,
                "elo_pred_margin": round(elo_margin, 3),
                "pred_margin": round(margin, 3),
                "pred_total": round(total, 3),
                "pred_home_points": round(float(scored["pred_home_points"]), 2),
                "pred_away_points": round(float(scored["pred_away_points"]), 2),
                "home_win_prob": round(home_win, 4),
                "actual_home_score": r.get("home_score"),
                "actual_away_score": r.get("away_score"),
                "high_confidence": abs(margin) >= float((weights.get("confidence") or {}).get("min_abs_margin", 10)),
            }
        )

        # update after completed game
        hs, aws = r.get("home_score"), r.get("away_score")
        if hs is None or aws is None:
            if day is not None:
                # upcoming: do not advance last_day until played
                pass
            continue

        hs_f, aws_f = float(hs), float(aws)
        exp_home = expected_score(rh + hfa, ra)
        actual = 0.5 if hs_f == aws_f else (1.0 if hs_f > aws_f else 0.0)
        mov = abs(hs_f - aws_f)
        mult = min(2.0, (mov + 1) ** 0.40)
        k_eff = k * (cupcake_k if r.get("is_cupcake") else 1.0)
        delta = k_eff * mult * (actual - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

        form_hist.setdefault(home, []).append((hs_f, aws_f))
        form_hist.setdefault(away, []).append((aws_f, hs_f))
        sim_hist.setdefault(home, []).append({"opp_elo": ra, "margin": hs_f - aws_f})
        sim_hist.setdefault(away, []).append({"opp_elo": rh, "margin": aws_f - hs_f})
        if day is not None:
            last_day[home] = day
            last_day[away] = day

    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    if both_fbs_eval and not out.is_empty() and "both_fbs" in out.columns:
        # keep all rows; evaluation filters separately
        pass
    return out


def backtest_margins(predictions: pl.DataFrame, *, both_fbs_only: bool = True) -> dict[str, float]:
    df = predictions.filter(
        pl.col("actual_home_score").is_not_null() & pl.col("actual_away_score").is_not_null()
    )
    if both_fbs_only and "both_fbs" in df.columns:
        df = df.filter(pl.col("both_fbs") == True)  # noqa: E712
    if df.is_empty():
        return {"n": 0}

    df = df.with_columns(
        actual_margin=pl.col("actual_home_score") - pl.col("actual_away_score"),
        pred_home_win=pl.col("pred_margin") > 0,
        actual_home_win=pl.col("actual_home_score") > pl.col("actual_away_score"),
    )
    mae = float((df["pred_margin"] - df["actual_margin"]).abs().mean())
    win_acc = float((df["pred_home_win"] == df["actual_home_win"]).mean())
    home_base = float(df["actual_home_win"].mean())
    return {
        "n": df.height,
        "mae_margin": round(mae, 3),
        "winner_accuracy": round(win_acc, 4),
        "home_baseline": round(home_base, 4),
    }


def gated_accuracy(
    predictions: pl.DataFrame,
    thresholds: list[float] | None = None,
    *,
    both_fbs_only: bool = True,
) -> dict[str, dict]:
    thresholds = thresholds or [7.0, 10.0, 14.0]
    df = predictions.filter(
        pl.col("actual_home_score").is_not_null() & pl.col("actual_away_score").is_not_null()
    )
    if both_fbs_only and "both_fbs" in df.columns:
        df = df.filter(pl.col("both_fbs") == True)  # noqa: E712
    out: dict[str, dict] = {}
    if df.is_empty():
        return {f"gate_{int(t)}": {"n": 0} for t in thresholds}

    actual_home = df["actual_home_score"] > df["actual_away_score"]
    pred_home = df["pred_margin"] > 0
    hits = (pred_home == actual_home).to_list()
    margins = df["pred_margin"].abs().to_list()
    for t in thresholds:
        sel = [h for h, m in zip(hits, margins) if m >= t]
        out[f"gate_{int(t)}"] = {
            "n": len(sel),
            "winner_accuracy": round(sum(sel) / len(sel), 4) if sel else None,
            "threshold": t,
        }
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
    thr = weights.get("edge_thresholds") or {}
    bets: list[dict] = []

    def maybe_add(market: str, label: str, prob: float, min_ev_key: str) -> None:
        prob = min(max(float(prob), 0.01), 0.99)
        price = -110
        ev = expected_value(prob, price)
        floor = float(thr.get(min_ev_key, 0.05))
        floor *= 0.75 if high_conf else 1.25
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
                    "pred_margin": margin,
                }
            )

    maybe_add("h2h", f"{home} ML", home_win, "h2h_min_ev")
    maybe_add("h2h", f"{away} ML", 1 - home_win, "h2h_min_ev")

    line = round(total * 2) / 2
    p_over = prob_over_line(total, float(weights.get("total_sigma", 12.0)), line)
    maybe_add("totals", f"Over {line}", p_over, "totals_min_ev")
    maybe_add("totals", f"Under {line}", 1 - p_over, "totals_min_ev")

    # Standard CFB spread proxy around model margin
    spread = -round(margin * 2) / 2 if abs(margin) >= 0.5 else -3.0
    # P(home covers -spread) when spread is home line (negative if favored)
    p_cover = 1.0 - normal_cdf(-spread, margin, sigma)
    maybe_add("spreads", f"{home} {spread:+g}", p_cover, "spreads_min_ev")
    maybe_add("spreads", f"{away} {-spread:+g}", 1 - p_cover, "spreads_min_ev")
    return bets


def predict_cfb_slate(
    *,
    season: int | None = None,
    week: int | None = None,
    hist_seasons: int = 3,
) -> dict[str, Any]:
    """Predict upcoming (or specified) FBS week slate."""
    ensure_dirs()
    weights = load_weights()
    save_weights(weights)
    season = season or current_cfb_season()
    seasons = list(range(max(2001, season - hist_seasons), season + 1))
    sched = load_schedules(seasons, regular_only=True, fbs_only=True, include_cupcakes=True)
    if sched.is_empty():
        return {
            "sport": "CFB",
            "season": season,
            "week": week,
            "games": [],
            "edges": [],
            "factors": factor_list(weights),
            "weights": weights,
            "data_source": data_source_used(),
        }

    preds = predict_games_walkforward(sched, weights)
    season_preds = preds.filter(pl.col("season") == season)
    if week is None:
        unfinished = season_preds.filter(pl.col("actual_home_score").is_null())
        if unfinished.is_empty():
            week = int(season_preds["week"].max()) if season_preds.height else 1
        else:
            week = int(unfinished["week"].min())
    week_preds = season_preds.filter(pl.col("week") == week)

    conf_min = float((weights.get("confidence") or {}).get("min_abs_margin", 10.0))
    sigma = float(weights.get("margin_sigma", 16.5))
    games_out: list[dict] = []
    for r in week_preds.iter_rows(named=True):
        # Betting board = unfinished both-FBS only (finished / cupcakes stay off edges)
        if r.get("actual_home_score") is not None:
            continue
        if r.get("is_cupcake"):
            continue
        margin = float(r["pred_margin"])
        total = float(r["pred_total"])
        home_win = float(r["home_win_prob"])
        high = abs(margin) >= conf_min
        home, away = str(r["home_team"]), str(r["away_team"])
        games_out.append(
            {
                "game_id": r.get("game_id"),
                "season": season,
                "week": week,
                "gameday": r.get("gameday"),
                "home_team": home,
                "away_team": away,
                "home_conference": r.get("home_conference"),
                "away_conference": r.get("away_conference"),
                "neutral_site": r.get("neutral_site"),
                "conference_game": r.get("conference_game"),
                "is_cupcake": r.get("is_cupcake"),
                "both_fbs": r.get("both_fbs"),
                "elo_home": r.get("elo_home"),
                "elo_away": r.get("elo_away"),
                "pred_margin": round(margin, 2),
                "pred_total": round(total, 2),
                "pred_home_points": r.get("pred_home_points"),
                "pred_away_points": r.get("pred_away_points"),
                "home_win_prob": round(home_win, 4),
                "high_confidence": high,
                "actual_home_score": r.get("actual_home_score"),
                "actual_away_score": r.get("actual_away_score"),
                "injuries_note": "CFB injury data sparse; model does not use injuries",
                "suggested_bets": _suggest_bets(
                    home, away, margin, total, home_win, sigma, weights, high_conf=high
                ),
            }
        )

    games_out.sort(key=lambda g: (str(g.get("gameday")), str(g.get("away_team"))))
    edges: list[dict] = []
    for g in games_out:
        edges.extend(g.get("suggested_bets") or [])

    if games_out:
        pl.DataFrame(
            [
                {k: v for k, v in g.items() if k not in {"suggested_bets"}}
                for g in games_out
            ]
        ).write_csv(OUTPUT_DIR / f"cfb_{season}_w{week}_predictions.csv")

    return {
        "sport": "CFB",
        "season": season,
        "week": week,
        "games": games_out,
        "edges": edges,
        "factors": factor_list(weights),
        "weights": weights,
        "data_source": data_source_used(),
    }


__all__ = [
    "predict_cfb_slate",
    "predict_games_walkforward",
    "backtest_margins",
    "gated_accuracy",
]
