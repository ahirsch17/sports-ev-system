from __future__ import annotations

from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.pricing import DevigMethod, devig_two_way


from sports_ev.features.time_utils import ensure_utc


OUT_STATUSES = {"out", "doubtful", "questionable"}


def build_injury_features(loader: FeatureDataLoader, ctx: FeatureContext) -> dict[str, float | None]:
    features: dict[str, float | None] = {}

    for side, team in (("home", ctx.game.home_team), ("away", ctx.game.away_team)):
        reports = loader.injury_reports(ctx, team)
        key_out = 0.0
        status_changed_close = 0.0

        for report in reports:
            report_ts = ensure_utc(report.report_timestamp)
            ctx.audit.add(
                f"{side}_injury_{report.player}",
                1.0 if report.status.lower() in OUT_STATUSES else 0.0,
                report_ts,
                f"injury_reports:{report.id}",
            )
            if report.status.lower() == "out":
                key_out = 1.0
            hours_before = (ctx.kickoff_time - report_ts).total_seconds() / 3600.0
            if 0 <= hours_before <= 24:
                status_changed_close = 1.0

        features[f"{side}_key_player_out"] = key_out
        features[f"{side}_injury_status_changed_24h"] = status_changed_close
        features[f"{side}_injury_report_count"] = float(len(reports))

    return features
