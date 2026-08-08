from __future__ import annotations

from statistics import mean

from sports_ev.features.mlb.park import park_factor_for_team
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.features.time_utils import ensure_utc

STAT_RUNS_SCORED = "runs_scored"
STAT_RUNS_ALLOWED = "runs_allowed"
STAT_HITS = "hits"
STAT_STARTER_INNINGS = "starter_innings"
STAT_STARTER_EARNED_RUNS = "starter_earned_runs"
STAT_BULLPEN_EARNED_RUNS = "bullpen_earned_runs"

CORE_STATS = (
    STAT_RUNS_SCORED,
    STAT_RUNS_ALLOWED,
    STAT_HITS,
    STAT_STARTER_INNINGS,
    STAT_STARTER_EARNED_RUNS,
    STAT_BULLPEN_EARNED_RUNS,
)


def _stats_by_key(history) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in history.stats:
        grouped.setdefault(row.stat_key, []).append(row.stat_value)
    return grouped


def _rolling_mean(values: list[float], window: int) -> float | None:
    if not values:
        return None
    return mean(values[:window])


def _season_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _opponent_adjusted(team_avg: float | None, opp_avg: float | None) -> float | None:
    if team_avg is None or opp_avg is None:
        return None
    return team_avg - opp_avg


def _starter_era_proxy(innings: list[float], earned_runs: list[float]) -> float | None:
    if not innings or not earned_runs:
        return None
    total_innings = sum(innings[:5])
    total_er = sum(earned_runs[:5])
    if total_innings <= 0:
        return None
    return (total_er / total_innings) * 9.0


def build_mlb_team_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    *,
    team: str,
    opponent: str,
    side: str,
    rolling_window: int = 10,
) -> dict[str, float | None]:
    history = loader.prior_team_stats(ctx, team)
    opp_history = loader.prior_team_stats(ctx, opponent)
    audit = ctx.audit

    for stat in history.stats:
        audit.add(
            f"{side}_{stat.stat_key}_hist",
            stat.stat_value,
            ensure_utc(stat.known_at),
            f"team_game_stats:{stat.game_id}",
        )

    team_grouped = _stats_by_key(history)
    opp_grouped = _stats_by_key(opp_history)
    features: dict[str, float | None] = {}

    for stat_key in CORE_STATS:
        team_vals = team_grouped.get(stat_key, [])
        opp_vals = opp_grouped.get(stat_key, [])
        season = _season_mean(team_vals)
        recent = _rolling_mean(team_vals, rolling_window)
        opp_season = _season_mean(opp_vals)
        adj = _opponent_adjusted(season, opp_season)

        features[f"{side}_{stat_key}_season"] = season
        features[f"{side}_{stat_key}_recent_{rolling_window}"] = recent
        features[f"{side}_{stat_key}_opp_adj"] = adj

    runs_scored = team_grouped.get(STAT_RUNS_SCORED, [])
    runs_allowed = team_grouped.get(STAT_RUNS_ALLOWED, [])
    if runs_scored and runs_allowed:
        run_diff = _season_mean([s - a for s, a in zip(runs_scored, runs_allowed)])
        features[f"{side}_run_diff_season"] = run_diff
    else:
        features[f"{side}_run_diff_season"] = None

    features[f"{side}_starter_era_proxy"] = _starter_era_proxy(
        team_grouped.get(STAT_STARTER_INNINGS, []),
        team_grouped.get(STAT_STARTER_EARNED_RUNS, []),
    )
    features[f"{side}_bullpen_er_recent"] = _rolling_mean(
        team_grouped.get(STAT_BULLPEN_EARNED_RUNS, []),
        rolling_window,
    )

    rest_days = loader.rest_days(ctx, team)
    if rest_days is not None:
        prior = loader.prior_games(ctx, team)
        if prior:
            audit.add(
                f"{side}_rest_days",
                rest_days,
                ensure_utc(prior[0].kickoff_time),
                f"prior_game:{prior[0].game_id}",
            )
    features[f"{side}_rest_days"] = rest_days
    features[f"{side}_is_home"] = 1.0 if side == "home" else 0.0

    return features


def build_mlb_matchup_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
) -> dict[str, float | None]:
    """Offense vs opposing pitching quality — encoded as features, not rules."""
    home_hist = loader.prior_team_stats(ctx, ctx.game.home_team)
    away_hist = loader.prior_team_stats(ctx, ctx.game.away_team)
    home_grouped = _stats_by_key(home_hist)
    away_grouped = _stats_by_key(away_hist)

    home_runs = _season_mean(home_grouped.get(STAT_RUNS_SCORED, []))
    away_runs = _season_mean(away_grouped.get(STAT_RUNS_SCORED, []))
    home_runs_allowed = _season_mean(home_grouped.get(STAT_RUNS_ALLOWED, []))
    away_runs_allowed = _season_mean(away_grouped.get(STAT_RUNS_ALLOWED, []))

    home_starter_era = _starter_era_proxy(
        home_grouped.get(STAT_STARTER_INNINGS, []),
        home_grouped.get(STAT_STARTER_EARNED_RUNS, []),
    )
    away_starter_era = _starter_era_proxy(
        away_grouped.get(STAT_STARTER_INNINGS, []),
        away_grouped.get(STAT_STARTER_EARNED_RUNS, []),
    )

    def offense_vs_pitching(offense_runs, opp_era):
        if offense_runs is None or opp_era is None:
            return None
        return offense_runs - (opp_era / 9.0)

    return {
        "matchup_home_offense_vs_away_pitching": offense_vs_pitching(home_runs, away_starter_era),
        "matchup_away_offense_vs_home_pitching": offense_vs_pitching(away_runs, home_starter_era),
        "matchup_runs_scored_delta": _opponent_adjusted(home_runs, away_runs),
        "matchup_runs_allowed_delta": _opponent_adjusted(home_runs_allowed, away_runs_allowed),
    }


def build_mlb_schedule_features(ctx: FeatureContext) -> dict[str, float | None]:
    return {
        "home_park_factor": park_factor_for_team(ctx.game.home_team),
    }
