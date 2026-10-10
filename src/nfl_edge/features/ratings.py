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
    min_periods: int = 3,
) -> pl.DataFrame:
    """
    Prior-to-game rolling offensive/defensive strength.
    offense = avg points scored; defense = avg points allowed (lower better).
    Also computes EMA-style recent form (last 3 margins).
    """
    df = team_games.sort(["team", "season", "week"])

    # shift so ratings are available before the game (no leakage)
    df = df.with_columns(
        prior_pf=pl.col("points_for").shift(1).over("team"),
        prior_pa=pl.col("points_against").shift(1).over("team"),
        prior_margin=pl.col("margin").shift(1).over("team"),
    )

    df = df.with_columns(
        off_rating=pl.col("prior_pf")
        .rolling_mean(window_size=window, min_samples=min_periods)
        .over("team"),
        def_rating=pl.col("prior_pa")
        .rolling_mean(window_size=window, min_samples=min_periods)
        .over("team"),
        form_ema=pl.col("prior_margin")
        .rolling_mean(window_size=3, min_samples=1)
        .over("team"),
        games_played=pl.col("prior_pf").cum_count().over("team"),
    )

    # season-to-date means as fallback early in season
    df = df.with_columns(
        season_off=(pl.col("prior_pf").cum_sum().over(["team", "season"]) / pl.col("prior_pf").cum_count().over(["team", "season"])),
        season_def=(pl.col("prior_pa").cum_sum().over(["team", "season"]) / pl.col("prior_pa").cum_count().over(["team", "season"])),
    )

    df = df.with_columns(
        off_rating=pl.coalesce([pl.col("off_rating"), pl.col("season_off"), pl.lit(22.0)]),
        def_rating=pl.coalesce([pl.col("def_rating"), pl.col("season_def"), pl.lit(22.0)]),
        form_ema=pl.coalesce([pl.col("form_ema"), pl.lit(0.0)]),
    )
    return df


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

    # prior rolling for pass/rush
    stats = stats.sort(["team", "season", "week"])
    for col in ["pass_yards", "rush_yards", "pass_epa", "rush_epa"]:
        if col in stats.columns:
            stats = stats.with_columns(
                pl.col(col).shift(1).rolling_mean(window_size=6, min_samples=2).over("team").alias(f"roll_{col}")
            )

    join_cols = ["season", "week", "team"]
    keep = [c for c in stats.columns if c.startswith("roll_") or c in join_cols]
    stats = stats.select(keep)
    return team_games.join(stats, on=join_cols, how="left")


def league_avg_points(team_games: pl.DataFrame) -> float:
    if team_games.is_empty():
        return 22.0
    return float(team_games["points_for"].mean())
