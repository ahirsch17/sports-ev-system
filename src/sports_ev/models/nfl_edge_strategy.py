"""Bridge: price the nfl_edge formula model inside the sports_ev pipeline.

nfl_edge predicts a home margin (Elo + points + weather + injuries + deep matchups +
situational). On its own it picks winners well (~64%) but is roughly break-even ATS, so
here it is blended with the margin implied by Pinnacle's devigged spread before pricing
against a soft book. Candidates flow through the normal sports_ev path: Kelly sizing,
paper bets with audit trail, CLV settlement and walk-forward backtests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from statistics import NormalDist
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from sports_ev.backtest.models import BetCandidate, BetSide
from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.pricing import DevigMethod, devig_two_way, expected_value, is_plus_ev

DEFAULT_SIGMA = 13.5
DEFAULT_MODEL_WEIGHT = 0.15
_EASTERN = ZoneInfo("America/New_York")
_STD = NormalDist()


def load_blend_params() -> tuple[float, float]:
    """(model_weight, sigma) from research/factor_weights.json (see research/fit_market_blend.py)."""
    try:
        from nfl_edge.features.weather import load_weights

        weights = load_weights()
    except Exception:  # noqa: BLE001
        return DEFAULT_MODEL_WEIGHT, DEFAULT_SIGMA
    blend = weights.get("market_blend") or {}
    return (
        float(blend.get("model_weight", DEFAULT_MODEL_WEIGHT)),
        float(weights.get("margin_sigma", DEFAULT_SIGMA)),
    )


class MarginPredictor(Protocol):
    def __call__(self, game: Game) -> float | None: ...


def side_cover_probs(mu: float, home_line: float, sigma: float = DEFAULT_SIGMA) -> tuple[float, float]:
    """P(home covers), P(away covers) given margin ~ N(mu, sigma), conditional on no push.

    Home covers when margin + home_line > 0. Whole-number lines carry push mass
    (margin exactly -home_line), approximated as the +/-0.5 band and excluded.
    """
    threshold = -float(home_line)
    if float(home_line).is_integer():
        win = 1.0 - _STD.cdf((threshold + 0.5 - mu) / sigma)
        loss = _STD.cdf((threshold - 0.5 - mu) / sigma)
    else:
        win = 1.0 - _STD.cdf((threshold - mu) / sigma)
        loss = 1.0 - win
    total = win + loss
    if total <= 0:
        return 0.5, 0.5
    return win / total, loss / total


def market_implied_margin(
    home_line: float, fair_home_prob: float, sigma: float = DEFAULT_SIGMA
) -> float:
    """Mean home margin that makes P(home covers home_line) equal the devigged price."""
    target = min(max(fair_home_prob, 1e-4), 1 - 1e-4)
    lo, hi = -80.0, 80.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if side_cover_probs(mid, home_line, sigma)[0] < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _local_date(kickoff: datetime) -> date:
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return kickoff.astimezone(_EASTERN).date()


@dataclass
class NflEdgeMarginPredictor:
    """Look up nfl_edge's predicted home margin for a sports_ev Game.

    Games are matched to the nflverse schedule by team abbreviations and local
    game date (Pinnacle fixtures carry no week). One model run is cached per
    (season, week).
    """

    upcoming_only: bool = False
    _schedules: dict[int, list[dict]] = field(default_factory=dict, repr=False)
    _margins: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict, repr=False)

    def _schedule_rows(self, season: int) -> list[dict]:
        if season not in self._schedules:
            from nfl_edge.ingest.nflverse import load_schedules

            sched = load_schedules([season])
            self._schedules[season] = sched.select(
                ["game_id", "season", "week", "gameday", "home_team", "away_team"]
            ).to_dicts()
        return self._schedules[season]

    def _locate(self, game: Game) -> tuple[int, int, str] | None:
        from nfl_edge.config import current_nfl_season
        from nfl_edge.edge.calculator import normalize_team

        kickoff_day = _local_date(game.kickoff_time)
        season = game.season or current_nfl_season(kickoff_day)
        home = normalize_team(game.home_team)
        away = normalize_team(game.away_team)
        for row in self._schedule_rows(int(season)):
            if row["home_team"] != home or row["away_team"] != away:
                continue
            gameday = row.get("gameday")
            if gameday:
                day = gameday if isinstance(gameday, date) else date.fromisoformat(str(gameday)[:10])
                if abs((day - kickoff_day).days) > 1:
                    continue
            return int(row["season"]), int(row["week"]), str(row["game_id"])
        return None

    def _week_margins(self, season: int, week: int) -> dict[str, float]:
        key = (season, week)
        if key not in self._margins:
            from nfl_edge.services.slate import model_week

            wm = model_week(season, week, upcoming_only=self.upcoming_only)
            self._margins[key] = (
                {}
                if wm is None
                else {
                    str(r["game_id"]): float(r["pred_margin"])
                    for r in wm.preds.iter_rows(named=True)
                    if r.get("pred_margin") is not None
                }
            )
        return self._margins[key]

    def __call__(self, game: Game) -> float | None:
        located = self._locate(game)
        if located is None:
            return None
        season, week, nflverse_id = located
        return self._week_margins(season, week).get(nflverse_id)


@dataclass
class NflEdgeSpreadStrategy:
    """NFL spreads priced from a blend of the nfl_edge margin and Pinnacle's implied margin.

    model_weight=1.0 trusts nfl_edge alone; 0.0 reduces to a line-aware market
    divergence strategy (handles soft lines that differ from Pinnacle's number).
    The fitted weight is small (~0.15): the model is a tilt on the sharp line, not a
    replacement for it.
    """

    predictor: MarginPredictor | Callable[[Game], float | None]
    name: str = "nfl_edge_blend_v1"
    soft_book: str = "draftkings"
    model_weight: float = DEFAULT_MODEL_WEIGHT
    sigma: float = DEFAULT_SIGMA
    require_sharp: bool = True
    devig_method: DevigMethod = DevigMethod.MULTIPLICATIVE
    market_type: str = "spread"

    def blended_margin(self, session: Session, game: Game, settings: Settings) -> dict | None:
        model_margin = self.predictor(game)
        if model_margin is None:
            return None
        loader = FeatureDataLoader(session)
        ctx = FeatureContext.for_game(game)
        pinnacle = loader.latest_odds_snapshot(
            ctx, book=settings.pinnacle_bookmaker, market_type="spread"
        )
        market_margin = None
        if (
            pinnacle is not None
            and pinnacle.line is not None
            and pinnacle.odds_home is not None
            and pinnacle.odds_away is not None
        ):
            fair_home, _ = devig_two_way(
                pinnacle.odds_home, pinnacle.odds_away, method=self.devig_method
            )
            market_margin = market_implied_margin(pinnacle.line, fair_home, self.sigma)
        if market_margin is None:
            if self.require_sharp:
                return None
            mu = float(model_margin)
        else:
            w = min(max(self.model_weight, 0.0), 1.0)
            mu = w * float(model_margin) + (1 - w) * market_margin
        return {"model_margin": float(model_margin), "market_margin": market_margin, "mu": mu}

    def generate_candidates(
        self, session: Session, game: Game, settings: Settings | None = None
    ) -> list[BetCandidate]:
        settings = settings or get_settings()
        loader = FeatureDataLoader(session)
        ctx = FeatureContext.for_game(game)
        soft = loader.latest_odds_snapshot(ctx, book=self.soft_book, market_type="spread")
        if soft is None or soft.line is None or soft.odds_home is None or soft.odds_away is None:
            return []

        blend = self.blended_margin(session, game, settings)
        if blend is None:
            return []
        p_home, p_away = side_cover_probs(blend["mu"], soft.line, self.sigma)

        candidates: list[BetCandidate] = []
        for side, prob, odds in (
            (BetSide.HOME, p_home, soft.odds_home),
            (BetSide.AWAY, p_away, soft.odds_away),
        ):
            if not is_plus_ev(prob, odds, min_edge_pct=settings.min_edge_pct):
                continue
            ev = expected_value(prob, odds, stake=1.0)
            candidates.append(
                BetCandidate(
                    game_id=game.game_id,
                    book=self.soft_book,
                    side=side,
                    american_odds=odds,
                    line=soft.line,
                    true_prob=prob,
                    edge_pct=ev.edge_pct,
                    decision_time=soft.captured_at,
                    market_type="spread",
                )
            )
        if len(candidates) <= 1:
            return candidates
        return [max(candidates, key=lambda c: c.edge_pct)]


def build_nfl_edge_strategy(
    *, soft_book: str = "draftkings", model_weight: float | None = None
) -> NflEdgeSpreadStrategy:
    fitted_weight, sigma = load_blend_params()
    return NflEdgeSpreadStrategy(
        predictor=NflEdgeMarginPredictor(),
        soft_book=soft_book,
        model_weight=fitted_weight if model_weight is None else model_weight,
        sigma=sigma,
    )
