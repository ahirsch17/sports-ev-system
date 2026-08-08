from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.registry import get_feature_pipeline
from sports_ev.models.mlb_labels import home_win_label


@dataclass(frozen=True)
class MlbLabeledRow:
    game_id: str
    kickoff_time: datetime
    features: dict[str, float | None]
    label: int


def build_mlb_labeled_rows(
    session: Session,
    *,
    season: int | None = None,
    before_kickoff: datetime | None = None,
    settings: Settings | None = None,
) -> list[MlbLabeledRow]:
    settings = settings or get_settings()
    pipeline = get_feature_pipeline(session, sport="mlb", settings=settings)

    query = session.query(Game).filter(
        Game.sport == "mlb",
        Game.home_score.isnot(None),
        Game.away_score.isnot(None),
    )
    if season is not None:
        query = query.filter(Game.season == season)
    if before_kickoff is not None:
        query = query.filter(Game.kickoff_time < before_kickoff)

    games = query.order_by(Game.kickoff_time.asc()).all()
    rows: list[MlbLabeledRow] = []

    for game in games:
        label = home_win_label(
            home_score=int(game.home_score),
            away_score=int(game.away_score),
        )
        if label is None:
            continue

        vector = pipeline.build_for_game(game.game_id, validate_leakage=True)
        rows.append(
            MlbLabeledRow(
                game_id=game.game_id,
                kickoff_time=game.kickoff_time,
                features=vector.features,
                label=label,
            )
        )

    return rows
