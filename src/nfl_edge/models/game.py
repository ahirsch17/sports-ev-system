"""Game-level spread/total predictions and simple backtest."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nfl_edge.features.matchups import build_matchup_frame
from nfl_edge.features.weather import load_weights
from nfl_edge.odds_math import margin_to_home_win_prob, normal_cdf, prob_over_line


@dataclass
class GamePrediction:
    game_id: str
    home_team: str
    away_team: str
    season: int
    week: int
    pred_margin: float
    pred_total: float
    home_win_prob: float
    pred_home_points: float
    pred_away_points: float
    pred_home_pass_yds: float
    pred_away_pass_yds: float
    pred_home_rush_yds: float
    pred_away_rush_yds: float


def predict_slate(
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
    player_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    weights = weights or load_weights()
    matchups = build_matchup_frame(
        schedules,
        team_stats,
        injury_features,
        weights,
        deep_features=deep_features,
        situational_features=situational_features,
        use_injuries=use_injuries,
        use_deep=use_deep,
        use_situational=use_situational,
        use_weather=use_weather,
        use_form=use_form,
        player_stats=player_stats,
    )
    sigma = float(weights.get("margin_sigma", 13.5))

    out = matchups.select(
        [
            c
            for c in [
                "game_id",
                "season",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "roof",
                "temp",
                "wind",
                "pred_margin",
                "pred_total",
                "pred_home_points",
                "pred_away_points",
                "pred_home_pass_yds",
                "pred_away_pass_yds",
                "pred_home_rush_yds",
                "pred_away_rush_yds",
                "home_pass_index",
                "away_pass_index",
                "home_te_mult",
                "away_te_mult",
                "home_script_note",
                "away_script_note",
                "home_score",
                "away_score",
                "deep_net_margin_adj",
                "home_sim_continuity",
                "away_sim_continuity",
                "home_sim_note",
                "away_sim_note",
                "home_mm_note",
                "away_mm_note",
                "home_sim_n",
                "away_sim_n",
                "situational_net_margin",
                "rest_margin_adj",
                "div_margin_adj",
                "rest_note",
                "div_note",
                "home_pace_note",
                "away_pace_note",
                "home_inj_note",
                "away_inj_note",
                "home_inj_points",
                "away_inj_points",
                "home_qb_out",
                "away_qb_out",
            ]
            if c in matchups.columns
        ]
    ).with_columns(
        home_win_prob=pl.col("pred_margin").map_elements(
            lambda m: round(margin_to_home_win_prob(float(m), sigma), 4),
            return_dtype=pl.Float64,
        )
    )
    rename = {}
    if "home_score" in out.columns:
        rename["home_score"] = "actual_home_score"
    if "away_score" in out.columns:
        rename["away_score"] = "actual_away_score"
    if rename:
        out = out.rename(rename)
    return out


def backtest_margins(predictions: pl.DataFrame) -> dict[str, float]:
    """Evaluate completed games only."""
    df = predictions.filter(
        pl.col("actual_home_score").is_not_null() & pl.col("actual_away_score").is_not_null()
    )
    if df.is_empty():
        return {"n": 0}

    df = df.with_columns(
        actual_margin=pl.col("actual_home_score") - pl.col("actual_away_score"),
        actual_total=pl.col("actual_home_score") + pl.col("actual_away_score"),
        pred_home_win=pl.col("pred_margin") > 0,
        actual_home_win=pl.col("actual_home_score") > pl.col("actual_away_score"),
    )
    mae_margin = float((df["pred_margin"] - df["actual_margin"]).abs().mean())
    mae_total = float((df["pred_total"] - df["actual_total"]).abs().mean())
    win_acc = float((df["pred_home_win"] == df["actual_home_win"]).mean())
    return {
        "n": df.height,
        "mae_margin": round(mae_margin, 3),
        "mae_total": round(mae_total, 3),
        "winner_accuracy": round(win_acc, 4),
    }


def cover_prob(pred_margin: float, spread_home: float, sigma: float = 13.5) -> float:
    """
    P(home covers) when home spread is spread_home (e.g. -3.5 means home favored).
    Home covers if (home - away) + spread_home > 0  i.e. margin > -spread_home
    """
    # margin ~ N(pred_margin, sigma); P(margin + spread > 0) = P(margin > -spread)
    return 1.0 - normal_cdf(-spread_home, pred_margin, sigma)


def total_side_probs(pred_total: float, line: float, sigma: float = 10.0) -> tuple[float, float]:
    over = prob_over_line(pred_total, sigma, line)
    return over, 1.0 - over
