"""Capture Pinnacle closing lines at kickoff for CLV tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.sports.registry import get_sport_config, list_sports


@dataclass
class ClosingCaptureResult:
    games_checked: int = 0
    games_captured: int = 0
    skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)


class ClosingLineCaptureService:
    """Freeze Pinnacle closing numbers on games that have started."""

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.loader = FeatureDataLoader(session)

    def capture_recent(self, *, lookback_hours: int = 48) -> ClosingCaptureResult:
        result = ClosingCaptureResult()
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=lookback_hours)

        games = (
            self.session.query(Game)
            .filter(
                Game.kickoff_time <= now,
                Game.kickoff_time >= window_start,
            )
            .order_by(Game.kickoff_time.desc())
            .all()
        )
        result.games_checked = len(games)

        for game in games:
            if game.closing_captured_at is not None:
                result.skipped_existing += 1
                continue
            try:
                if self._capture_game(game):
                    result.games_captured += 1
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{game.game_id}: {exc}")

        return result

    def _capture_game(self, game: Game) -> bool:
        config = get_sport_config(game.sport, self.settings)
        ctx = FeatureContext.for_game(game)
        market_type = config.default_market
        pinnacle = self.loader.latest_odds_snapshot(
            ctx,
            book=self.settings.pinnacle_bookmaker,
            market_type=market_type,
        )
        if pinnacle is None:
            return False

        game.pinnacle_closing_home_odds = pinnacle.odds_home
        game.pinnacle_closing_away_odds = pinnacle.odds_away
        game.pinnacle_closing_spread = pinnacle.line if market_type == "spread" else None
        game.closing_captured_at = datetime.now(timezone.utc)
        return True


def capture_closing_lines_for_all_sports(
    session: Session,
    *,
    settings: Settings | None = None,
    lookback_hours: int = 48,
) -> ClosingCaptureResult:
    service = ClosingLineCaptureService(session, settings=settings)
    return service.capture_recent(lookback_hours=lookback_hours)
