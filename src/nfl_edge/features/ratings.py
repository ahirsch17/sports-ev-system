"""Team offensive/defensive ratings from game results and EPA proxies."""

from __future__ import annotations

import polars as pl


def _score_frame(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game with points for/against and margin."""
    completed = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    home = completed.select(
        pl.col("game_id"),
        pl.col("season"),
        pl.col("week"),
        pl.col("gameday"),
        team=pl.col("home_team"),
        opp=pl.col("away_team"),
        is_home=pl.lit(1),
        points_for=pl.col("home_score"),
        points_against=pl.col("away_score"),
        rest=pl.col("home_rest") if "home_rest" in completed.columns else pl.lit(7),
    )
    away = completed.select(
        pl.col("game_id"),
        pl.col("season"),
        pl.col("week"),
        pl.col("gameday"),
        team=pl.col("away_team"),
        opp=pl.col("home_team"),
        is_home=pl.lit(0),
        points_for=pl.col("away_score"),
        points_against=pl.col("home_score"),
        rest=pl.col("away_rest") if "away_rest" in completed.columns else pl.lit(7),
    )
    return pl.concat([home, away], how="diagonal_relaxed").with_columns(
        margin=pl.col("points_for") - pl.col("points_against")
    )


def build_team_game_log(schedules: pl.DataFrame) -> pl.DataFrame:
    return _score_frame(schedules).sort(["season", "week", "team"])


def rolling_team_ratings(
    team_games: pl.DataFrame,
    *,
    window: int = 8,
    min_samples: int = 3,
) -> pl.DataFrame:
    """
    Rolling offensive/defensive strength per team-game.
    offense = avg points scored; defense = avg points allowed (lower better).
    Also computes EMA-style recent form (last 3 margins).

    `off_rating`/`def_rating`/`form_ema` are as of kickoff of that row's game
    (no leakage). `post_*` columns include that game's result: they are the
    rating a team carries into its *next* game.
    """
    df = team_games.sort(["team", "season", "week"])

    # shift so ratings are available before the game (no leakage)
    df = df.with_columns(
        prior_pf=pl.col("points_for").shift(1).over("team"),
        prior_pa=pl.col("points_against").shift(1).over("team"),
        prior_margin=pl.col("margin").shift(1).over("team"),
    )

    def _ratings(pf: str, pa: str, margin: str, prefix: str) -> list[pl.Expr]:
        by_season = ["team", "season"]
        return [
            pl.coalesce(
                pl.col(pf).rolling_mean(window_size=window, min_samples=min_samples).over("team"),
                pl.col(pf).cum_sum().over(by_season) / pl.col(pf).cum_count().over(by_season),
                pl.lit(22.0),
            ).alias(f"{prefix}off_rating"),
            pl.coalesce(
                pl.col(pa).rolling_mean(window_size=window, min_samples=min_samples).over("team"),
                pl.col(pa).cum_sum().over(by_season) / pl.col(pa).cum_count().over(by_season),
                pl.lit(22.0),
            ).alias(f"{prefix}def_rating"),
            pl.coalesce(
                pl.col(margin).rolling_mean(window_size=3, min_samples=1).over("team"),
                pl.lit(0.0),
            ).alias(f"{prefix}form_ema"),
        ]

    return df.with_columns(
        *_ratings("prior_pf", "prior_pa", "prior_margin", ""),
        *_ratings("points_for", "points_against", "margin", "post_"),
        games_played=pl.col("prior_pf").cum_count().over("team"),
    )


def attach_pass_rush_splits(team_games: pl.DataFrame, team_stats: pl.DataFrame) -> pl.DataFrame:
    """Join weekly team pass/rush yards if team_stats available."""
    if team_stats.is_empty():
        return team_games.with_columns(
            pass_yards=pl.lit(None, dtype=pl.Float64),
            rush_yards=pl.lit(None, dtype=pl.Float64),
            pass_epa=pl.lit(None, dtype=pl.Float64),
            rush_epa=pl.lit(None, dtype=pl.Float64),
        )

    # nflverse team stats column names vary; detect flexibly
    cols = set(team_stats.columns)
    team_col = "team" if "team" in cols else ("recent_team" if "recent_team" in cols else None)
    if team_col is None or "season" not in cols or "week" not in cols:
        return team_games

    select_cols = [team_col, "season", "week"]
    renames: dict[str, str] = {}
    for src, dst in [
        ("passing_yards", "pass_yards"),
        ("rushing_yards", "rush_yards"),
        ("passing_epa", "pass_epa"),
        ("rushing_epa", "rush_epa"),
        ("pass_yards", "pass_yards"),
        ("rush_yards", "rush_yards"),
    ]:
        if src in cols and dst not in renames.values():
            select_cols.append(src)
            if src != dst:
                renames[src] = dst

    stats = team_stats.select([c for c in select_cols if c in team_stats.columns])
    if renames:
        stats = stats.rename(renames)
    if team_col != "team":
        stats = stats.rename({team_col: "team"})
    stats = stats.unique(subset=["season", "week", "team"], keep="first")

    # Yards the opponent gained that week are the yards this defense allowed.
    if "opp" in team_games.columns and "pass_yards" in stats.columns:
        opp_cols = [
            "season",
            "week",
            pl.col("team").alias("opp"),
            pl.col("pass_yards").alias("pass_allowed"),
        ]
        if "rush_yards" in stats.columns:
            opp_cols.append(pl.col("rush_yards").alias("rush_allowed"))
        faced = (
            team_games.select(["season", "week", "team", "opp"])
            .unique(subset=["season", "week", "team"], keep="first")
            .join(stats.select(opp_cols), on=["season", "week", "opp"], how="left")
        )
        keep_faced = ["season", "week", "team", "pass_allowed"]
        if "rush_allowed" in faced.columns:
            keep_faced.append("rush_allowed")
        stats = stats.join(faced.select(keep_faced), on=["season", "week", "team"], how="left")

    # prior rolling for pass/rush
    stats = stats.sort(["team", "season", "week"])
    for col in ["pass_yards", "rush_yards", "pass_epa", "rush_epa", "pass_allowed", "rush_allowed"]:
        if col in stats.columns:
            stats = stats.with_columns(
                pl.col(col).shift(1).rolling_mean(window_size=6, min_samples=2).over("team").alias(f"roll_{col}"),
                pl.col(col).rolling_mean(window_size=6, min_samples=2).over("team").alias(f"post_roll_{col}"),
            )

    join_cols = ["season", "week", "team"]
    keep = [c for c in stats.columns if c.startswith(("roll_", "post_roll_")) or c in join_cols]
    stats = stats.select(keep)
    return team_games.join(stats, on=join_cols, how="left")


def league_avg_points(team_games: pl.DataFrame) -> float:
    if team_games.is_empty():
        return 22.0
    return float(team_games["points_for"].mean())
