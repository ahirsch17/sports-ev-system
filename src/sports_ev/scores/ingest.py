from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.scores.espn import EspnScoreClient
from sports_ev.softbooks.matching import find_game_for_matchup
from sports_ev.sports.registry import get_sport_config


@dataclass
class ScoreIngestResult:
    scores_seen: int = 0
    games_updated: int = 0
    unmatched: int = 0
    errors: list[str] = field(default_factory=list)


class ScoreIngestService:
    """Pull final scores from ESPN and update Game rows for settlement."""

    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        client: EspnScoreClient | None = None,
    ):
        self.session = session
        self.settings = settings or get_settings()
        self.client = client or EspnScoreClient()
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def ingest_sport(
        self,
        sport: str,
        *,
        lookback_days: int = 3,
    ) -> ScoreIngestResult:
        config = get_sport_config(sport, self.settings)
        result = ScoreIngestResult()
        today = date.today()

        for offset in range(lookback_days + 1):
            on_date = today - timedelta(days=offset)
            try:
                scores = self.client.fetch_completed_scores(
                    config.espn_scoreboard_path,
                    on_date=on_date,
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{sport} {on_date}: {exc}")
                continue

            for score in scores:
                result.scores_seen += 1
                game = find_game_for_matchup(
                    self.session,
                    sport=config.sport,
                    home_team=score.home_team,
                    away_team=score.away_team,
                    kickoff_time=None,
                    event_date=score.event_date,
                )
                if game is None:
                    result.unmatched += 1
                    continue

                if (
                    game.home_score == score.home_score
                    and game.away_score == score.away_score
                ):
                    continue

                game.home_score = score.home_score
                game.away_score = score.away_score
                result.games_updated += 1

        self.session.flush()
        return result

    def ingest_all(self, *, lookback_days: int = 3) -> dict[str, ScoreIngestResult]:
        from sports_ev.sports.registry import list_sports

        return {
            sport: self.ingest_sport(sport, lookback_days=lookback_days)
            for sport in list_sports()
        }
