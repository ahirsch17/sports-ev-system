from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from sports_ev.db.models import Game, InjuryReport, OddsSnapshot, TeamGameStat
from sports_ev.features.leakage import FeatureAudit
from sports_ev.features.time_utils import ensure_utc


@dataclass(frozen=True)
class FeatureContext:
    game: Game
    kickoff_time: datetime
    audit: FeatureAudit

    @classmethod
    def for_game(cls, game: Game) -> FeatureContext:
        kickoff = ensure_utc(game.kickoff_time)
        return cls(game=game, kickoff_time=kickoff, audit=FeatureAudit(kickoff_time=kickoff))


@dataclass
class TeamHistory:
    team: str
    stats: list[TeamGameStat]


class FeatureDataLoader:
    def __init__(self, session: Session):
        self.session = session

    def prior_team_stats(self, ctx: FeatureContext, team: str) -> TeamHistory:
        rows = (
            self.session.query(TeamGameStat)
            .join(Game, TeamGameStat.game_id == Game.game_id)
            .filter(
                TeamGameStat.team == team,
                TeamGameStat.known_at < ctx.kickoff_time,
                Game.kickoff_time < ctx.kickoff_time,
            )
            .order_by(TeamGameStat.known_at.desc())
            .all()
        )
        return TeamHistory(team=team, stats=rows)

    def latest_odds_snapshot(
        self,
        ctx: FeatureContext,
        *,
        book: str,
        market_type: str = "spread",
    ) -> OddsSnapshot | None:
        return (
            self.session.query(OddsSnapshot)
            .filter(
                OddsSnapshot.game_id == ctx.game.game_id,
                OddsSnapshot.book == book,
                OddsSnapshot.market_type == market_type,
                OddsSnapshot.captured_at < ctx.kickoff_time,
            )
            .order_by(OddsSnapshot.captured_at.desc())
            .first()
        )

    def odds_snapshots_before_kickoff(
        self,
        ctx: FeatureContext,
        *,
        book: str,
        market_type: str = "spread",
    ) -> list[OddsSnapshot]:
        return (
            self.session.query(OddsSnapshot)
            .filter(
                OddsSnapshot.game_id == ctx.game.game_id,
                OddsSnapshot.book == book,
                OddsSnapshot.market_type == market_type,
                OddsSnapshot.captured_at < ctx.kickoff_time,
            )
            .order_by(OddsSnapshot.captured_at.asc())
            .all()
        )

    def injury_reports(self, ctx: FeatureContext, team: str) -> list[InjuryReport]:
        return (
            self.session.query(InjuryReport)
            .filter(
                InjuryReport.game_id == ctx.game.game_id,
                InjuryReport.team == team,
                InjuryReport.report_timestamp < ctx.kickoff_time,
            )
            .order_by(InjuryReport.report_timestamp.desc())
            .all()
        )

    def prior_games(self, ctx: FeatureContext, team: str) -> list[Game]:
        return (
            self.session.query(Game)
            .filter(
                Game.sport == ctx.game.sport,
                Game.kickoff_time < ctx.kickoff_time,
                (Game.home_team == team) | (Game.away_team == team),
            )
            .order_by(Game.kickoff_time.desc())
            .all()
        )

    def game_team_stats(self, ctx: FeatureContext, team: str) -> list[TeamGameStat]:
        """Stats recorded for this specific game (e.g. probable starter lines)."""
        return (
            self.session.query(TeamGameStat)
            .filter(
                TeamGameStat.game_id == ctx.game.game_id,
                TeamGameStat.team == team,
                TeamGameStat.known_at < ctx.kickoff_time,
            )
            .order_by(TeamGameStat.known_at.desc())
            .all()
        )

    def rest_days(self, ctx: FeatureContext, team: str) -> float | None:
        games = self.prior_games(ctx, team)
        if not games:
            return None
        last = games[0].kickoff_time
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta = ctx.kickoff_time - last
        return delta.total_seconds() / 86400.0

    def is_divisional(self, ctx: FeatureContext) -> bool:
        """Placeholder: true when teams share a division token in name (stub until schedule metadata)."""
        home_tokens = set(ctx.game.home_team.lower().split())
        away_tokens = set(ctx.game.away_team.lower().split())
        # Without real division data, use no false positives — always 0 unless configured
        return False

    def days_since_stat(self, ctx: FeatureContext, stat: TeamGameStat) -> float:
        known = stat.known_at
        if known.tzinfo is None:
            known = known.replace(tzinfo=timezone.utc)
        return (ctx.kickoff_time - known).total_seconds() / 86400.0
