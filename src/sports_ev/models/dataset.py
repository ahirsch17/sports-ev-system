from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.pipeline import FeaturePipeline
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.models.labels import home_cover_label


@dataclass(frozen=True)
class LabeledRow:
    game_id: str
    kickoff_time: datetime
    features: dict[str, float | None]
    label: int
    spread_line: float


def _decision_spread_line(session: Session, game: Game, settings: Settings) -> float | None:
    loader = FeatureDataLoader(session)
    ctx = FeatureContext.for_game(game)
    pinnacle = loader.latest_odds_snapshot(
        ctx, book=settings.pinnacle_bookmaker, market_type="spread"
    )
    if pinnacle and pinnacle.line is not None:
        return pinnacle.line
    soft = loader.latest_odds_snapshot(ctx, book="draftkings", market_type="spread")
    return soft.line if soft else None


def build_labeled_rows(
    session: Session,
    *,
    sport: str = "nfl",
    season: int | None = None,
    before_kickoff: datetime | None = None,
    settings: Settings | None = None,
) -> list[LabeledRow]:
    """Build time-ordered labeled rows using only pre-kickoff features."""
    settings = settings or get_settings()
    pipeline = FeaturePipeline(session, settings)

    query = session.query(Game).filter(
        Game.sport == sport,
        Game.home_score.isnot(None),
        Game.away_score.isnot(None),
    )
    if season is not None:
        query = query.filter(Game.season == season)
    if before_kickoff is not None:
        query = query.filter(Game.kickoff_time < before_kickoff)

    games = query.order_by(Game.kickoff_time.asc()).all()
    rows: list[LabeledRow] = []

    for game in games:
        spread_line = _decision_spread_line(session, game, settings)
        if spread_line is None:
            continue

        label = home_cover_label(
            home_score=int(game.home_score),
            away_score=int(game.away_score),
            home_line=spread_line,
        )
        if label is None:
            continue

        vector = pipeline.build_for_game(game.game_id, validate_leakage=True)
        rows.append(
            LabeledRow(
                game_id=game.game_id,
                kickoff_time=game.kickoff_time,
                features=vector.features,
                label=label,
                spread_line=spread_line,
            )
        )

    return rows


def feature_names_from_rows(rows: list[LabeledRow]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row.features.keys())
    return sorted(names)


def rows_to_matrix(rows: list[LabeledRow], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [
                float(row.features[name]) if row.features.get(name) is not None else np.nan
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.array([row.label for row in rows], dtype=np.int32)
    return x, y


def row_to_vector(row_features: dict[str, float | None], feature_names: list[str]) -> np.ndarray:
    values = []
    for name in feature_names:
        val = row_features.get(name)
        values.append(float(val) if val is not None else np.nan)
    return np.array([values], dtype=np.float64)
