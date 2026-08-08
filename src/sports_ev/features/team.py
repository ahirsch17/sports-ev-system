from __future__ import annotations

from statistics import mean

from sports_ev.features.sources import FeatureContext, FeatureDataLoader, TeamHistory
from sports_ev.features.time_utils import ensure_utc


STAT_EPA_OFFENSE = "epa_offense"
STAT_EPA_DEFENSE = "epa_defense"
STAT_EPA_EARLY_DOWN = "epa_early_down"
STAT_EPA_RED_ZONE = "epa_red_zone"
STAT_EPA_THIRD_DOWN = "epa_third_down"
STAT_SUCCESS_RATE = "success_rate"
STAT_WR_TARGET_SHARE = "wr_target_share"
STAT_WR_EPA = "wr_epa"
STAT_PASS_DEF_EPA = "pass_def_epa"


def _stats_by_key(history: TeamHistory) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in history.stats:
        grouped.setdefault(row.stat_key, []).append(row.stat_value)
    return grouped


def _rolling_mean(values: list[float], window: int) -> float | None:
    if not values:
        return None
    sample = values[:window]
    return mean(sample)


def _season_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _opponent_adjusted(team_avg: float | None, opp_avg: float | None, league_avg: float = 0.0) -> float | None:
    if team_avg is None or opp_avg is None:
        return None
    return team_avg - opp_avg + league_avg


def build_team_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    *,
    team: str,
    opponent: str,
    side: str,
    rolling_window: int = 4,
) -> dict[str, float | None]:
    history = loader.prior_team_stats(ctx, team)
    opp_history = loader.prior_team_stats(ctx, opponent)
    audit = ctx.audit

    for stat in history.stats:
        known_at = ensure_utc(stat.known_at)
        audit.add(
            f"{side}_{stat.stat_key}_hist",
            stat.stat_value,
            known_at,
            f"team_game_stats:{stat.game_id}",
        )

    team_grouped = _stats_by_key(history)
    opp_grouped = _stats_by_key(opp_history)

    features: dict[str, float | None] = {}

    for stat_key in (
        STAT_EPA_OFFENSE,
        STAT_EPA_DEFENSE,
        STAT_EPA_EARLY_DOWN,
        STAT_EPA_RED_ZONE,
        STAT_EPA_THIRD_DOWN,
        STAT_SUCCESS_RATE,
    ):
        team_vals = team_grouped.get(stat_key, [])
        opp_vals = opp_grouped.get(stat_key, [])
        season = _season_mean(team_vals)
        recent = _rolling_mean(team_vals, rolling_window)
        opp_season = _season_mean(opp_vals)
        adj = _opponent_adjusted(season, opp_season)

        features[f"{side}_{stat_key}_season"] = season
        features[f"{side}_{stat_key}_recent_{rolling_window}"] = recent
        features[f"{side}_{stat_key}_opp_adj"] = adj

    rest_days = loader.rest_days(ctx, team)
    if rest_days is not None:
        # Rest is derived from prior game kickoff — audit with that timestamp
        prior = loader.prior_games(ctx, team)
        if prior:
            known_at = ensure_utc(prior[0].kickoff_time)
            audit.add(f"{side}_rest_days", rest_days, known_at, f"prior_game:{prior[0].game_id}")

    features[f"{side}_rest_days"] = rest_days
    features[f"{side}_is_home"] = 1.0 if side == "home" else 0.0

    return features


def build_matchup_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
) -> dict[str, float | None]:
    """WR efficiency vs opposing pass defense — encoded as a feature, not a rule."""
    home_hist = loader.prior_team_stats(ctx, ctx.game.home_team)
    away_hist = loader.prior_team_stats(ctx, ctx.game.away_team)

    def latest_stat(history: TeamHistory, key: str) -> float | None:
        for stat in history.stats:
            if stat.stat_key == key:
                known_at = ensure_utc(stat.known_at)
                ctx.audit.add(
                    f"matchup_{key}",
                    stat.stat_value,
                    known_at,
                    f"team_game_stats:{stat.game_id}",
                )
                return stat.stat_value
        return None

    home_wr_epa = latest_stat(home_hist, STAT_WR_EPA)
    home_wr_share = latest_stat(home_hist, STAT_WR_TARGET_SHARE)
    away_pass_def = latest_stat(away_hist, STAT_PASS_DEF_EPA)

    away_wr_epa = latest_stat(away_hist, STAT_WR_EPA)
    away_wr_share = latest_stat(away_hist, STAT_WR_TARGET_SHARE)
    home_pass_def = latest_stat(home_hist, STAT_PASS_DEF_EPA)

    def wr_vs_pass_def(wr_epa, wr_share, pass_def):
        if wr_epa is None or wr_share is None or pass_def is None:
            return None
        return wr_epa * wr_share - pass_def

    return {
        "matchup_home_wr_vs_away_pass_def": wr_vs_pass_def(home_wr_epa, home_wr_share, away_pass_def),
        "matchup_away_wr_vs_home_pass_def": wr_vs_pass_def(away_wr_epa, away_wr_share, home_pass_def),
    }


def build_schedule_features(loader: FeatureDataLoader, ctx: FeatureContext) -> dict[str, float]:
    return {
        "is_divisional": 1.0 if loader.is_divisional(ctx) else 0.0,
    }
