from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from sports_ev.backtest.models import BetSide
from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.registry import get_feature_pipeline
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.models.mlb_artifact import MlbMoneylineModel
from sports_ev.models.artifact import NflSpreadModel
from sports_ev.paper.kelly import kelly_fraction
from sports_ev.pricing import expected_value, american_to_implied_prob
from sports_ev.sports.registry import get_sport_config, list_sports


@dataclass(frozen=True)
class ModelPick:
    game_id: str
    sport: str
    matchup: str
    kickoff_time: datetime
    side: str
    side_label: str
    american_odds: int
    line_taken: float | None
    market_type: str
    model_prob: float
    market_implied_prob: float
    edge_pct: float
    suggested_stake_pct: float
    model_version: str
    meets_plus_ev_threshold: bool


def pick_is_bettable(pick: ModelPick, settings: Settings | None = None) -> bool:
    """True when edge, Kelly stake, and market data support a real paper bet."""
    settings = settings or get_settings()
    if pick.edge_pct < settings.min_edge_pct or pick.edge_pct <= 0:
        return False
    if pick.suggested_stake_pct <= 0:
        return False
    if pick.market_type == "spread" and pick.line_taken is None:
        return False
    return True


def sport_has_model(sport: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    try:
        _load_model(config.sport, config.default_market, settings)
        return True
    except (FileNotFoundError, ValueError):
        return False


def _load_model(sport: str, market_type: str, settings: Settings):
    if sport == "mlb" and market_type == "moneyline":
        path = settings.mlb_model_artifact_path
        if not Path(path).exists():
            raise FileNotFoundError(f"MLB model artifact not found: {path}")
        return MlbMoneylineModel.load(path)
    if sport == "nfl" and market_type == "spread":
        path = settings.model_artifact_path
        if not Path(path).exists():
            raise FileNotFoundError(f"NFL model artifact not found: {path}")
        return NflSpreadModel.load(path)
    raise ValueError(f"No model configured for {sport}/{market_type}")


def get_model_picks(
    session: Session,
    *,
    sport: str,
    soft_book: str = "draftkings",
    days_ahead: int = 7,
    settings: Settings | None = None,
) -> tuple[list[ModelPick], list[str]]:
    """Best model-side pick per upcoming game, even when below the +EV threshold."""
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    market_type = config.default_market
    errors: list[str] = []
    picks: list[ModelPick] = []

    try:
        model = _load_model(config.sport, market_type, settings)
    except (FileNotFoundError, ValueError) as exc:
        return [], [str(exc)]

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    games = (
        session.query(Game)
        .filter(
            Game.sport == config.sport,
            Game.kickoff_time > now,
            Game.kickoff_time <= end,
        )
        .order_by(Game.kickoff_time.asc())
        .all()
    )

    pipeline = get_feature_pipeline(session, sport=config.sport, settings=settings)
    loader = FeatureDataLoader(session)

    for game in games:
        ctx = FeatureContext.for_game(game)
        soft = loader.latest_odds_snapshot(ctx, book=soft_book, market_type=market_type)
        if soft is None or soft.odds_home is None or soft.odds_away is None:
            errors.append(f"No {soft_book} lines for {game.away_team} @ {game.home_team}")
            continue

        try:
            vector = pipeline.build_for_game(game.game_id, validate_leakage=True)
            if config.sport == "mlb":
                prediction = model.predict_one(vector.features, compute_shap=False)
                home_prob = prediction.home_win_prob
                away_prob = prediction.away_win_prob
            else:
                prediction = model.predict_one(vector.features, compute_shap=False)
                home_prob = prediction.home_cover_prob
                away_prob = 1.0 - home_prob
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Feature/model error for {game.away_team} @ {game.home_team}: {exc}")
            continue

        home_ev = expected_value(home_prob, soft.odds_home, stake=1.0)
        away_ev = expected_value(away_prob, soft.odds_away, stake=1.0)

        if home_ev.edge_pct >= away_ev.edge_pct:
            side = BetSide.HOME.value
            side_label = game.home_team
            odds = int(soft.odds_home)
            model_prob = home_prob
            market_implied = american_to_implied_prob(soft.odds_home)
            edge_pct = home_ev.edge_pct
        else:
            side = BetSide.AWAY.value
            side_label = game.away_team
            odds = int(soft.odds_away)
            model_prob = away_prob
            market_implied = american_to_implied_prob(soft.odds_away)
            edge_pct = away_ev.edge_pct

        line_taken: float | None = None
        if market_type == "spread":
            if soft.line is None:
                errors.append(f"No spread line for {game.away_team} @ {game.home_team}")
                continue
            line_taken = float(soft.line) if side == BetSide.HOME.value else float(-soft.line)

        stake_pct = kelly_fraction(
            model_prob,
            odds,
            fraction=settings.kelly_fraction,
            max_stake_pct=settings.max_stake_pct,
        )
        meets = (
            edge_pct >= settings.min_edge_pct
            and edge_pct > 0
            and stake_pct > 0
            and (market_type != "spread" or line_taken is not None)
        )
        picks.append(
            ModelPick(
                game_id=game.game_id,
                sport=config.sport,
                matchup=f"{game.away_team} @ {game.home_team}",
                kickoff_time=game.kickoff_time,
                side=side,
                side_label=side_label,
                american_odds=odds,
                line_taken=line_taken,
                market_type=market_type,
                model_prob=model_prob,
                market_implied_prob=market_implied,
                edge_pct=edge_pct,
                suggested_stake_pct=stake_pct,
                model_version=getattr(model, "model_version", "model"),
                meets_plus_ev_threshold=meets,
            )
        )

    return picks, errors


def get_all_sports_picks_24h(
    session: Session,
    *,
    soft_book: str = "draftkings",
    settings: Settings | None = None,
) -> tuple[list[ModelPick], list[str]]:
    """Model picks for every sport with kickoff in the next 24 hours."""
    settings = settings or get_settings()
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=24)
    all_picks: list[ModelPick] = []
    all_errors: list[str] = []

    for sport in list_sports():
        if not sport_has_model(sport, settings):
            continue
        picks, errors = get_model_picks(
            session,
            sport=sport,
            soft_book=soft_book,
            days_ahead=7,
            settings=settings,
        )
        for pick in picks:
            kickoff = pick.kickoff_time
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            if now < kickoff <= end:
                all_picks.append(pick)
        all_errors.extend(errors)

    all_picks.sort(key=lambda p: p.kickoff_time)
    return all_picks, all_errors
