from __future__ import annotations

from datetime import datetime

from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.features.time_utils import ensure_utc

PROBABLE_STAT_KEYS = (
    "probable_starter_era",
    "probable_starter_whip",
    "probable_starter_k9",
)


def _starter_stats_for_game(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    team: str,
) -> tuple[dict[str, float], dict[str, datetime]]:
    rows = loader.game_team_stats(ctx, team)
    values = {row.stat_key: row.stat_value for row in rows if row.stat_key in PROBABLE_STAT_KEYS}
    known_at = {
        row.stat_key: ensure_utc(row.known_at)
        for row in rows
        if row.stat_key in PROBABLE_STAT_KEYS
    }
    return values, known_at


def build_probable_pitcher_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
    *,
    home_team: str,
    away_team: str,
) -> dict[str, float | None]:
    home_stats, home_known = _starter_stats_for_game(loader, ctx, home_team)
    away_stats, away_known = _starter_stats_for_game(loader, ctx, away_team)

    features: dict[str, float | None] = {}
    for side, stats, known_map in (
        ("home", home_stats, home_known),
        ("away", away_stats, away_known),
    ):
        for key in PROBABLE_STAT_KEYS:
            short = key.replace("probable_starter_", "")
            value = stats.get(key)
            features[f"{side}_probable_{short}"] = value
            if value is not None:
                ctx.audit.add(
                    f"{side}_probable_{short}",
                    value,
                    known_map[key],
                    f"team_game_stats:{ctx.game.game_id}",
                )

    home_era = home_stats.get("probable_starter_era")
    away_era = away_stats.get("probable_starter_era")
    if home_era is not None and away_era is not None:
        features["matchup_probable_era_delta"] = away_era - home_era
    else:
        features["matchup_probable_era_delta"] = None

    return features


def build_mlb_pitcher_features(
    loader: FeatureDataLoader,
    ctx: FeatureContext,
) -> dict[str, float | None]:
    return build_probable_pitcher_features(
        loader,
        ctx,
        home_team=ctx.game.home_team,
        away_team=ctx.game.away_team,
    )
