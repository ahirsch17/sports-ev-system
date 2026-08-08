from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game, OddsSnapshot, PaperBet
from sports_ev.features.sources import FeatureContext, FeatureDataLoader
from sports_ev.pricing.devig import devig_two_way
from sports_ev.pricing.ev import expected_value
from sports_ev.pricing.odds import american_to_implied_prob


@dataclass(frozen=True)
class OddsQuote:
    odds_home: int | None
    odds_away: int | None
    line: float | None
    captured_at: datetime | None
    book: str


@dataclass
class ManualBetRequest:
    game_id: str
    book: str
    market_type: str
    side: str
    odds_taken: int
    line_taken: float | None = None
    stake_pct: float | None = None
    model_version: str | None = None
    edge_pct: float | None = None
    audit_extra: dict | None = None


@dataclass
class PlacementResult:
    success: bool
    message: str
    bet_id: int | None = None
    duplicate: bool = False
    updated: bool = False


def latest_odds_quote(
    session: Session,
    *,
    game_id: str,
    book: str,
    market_type: str,
) -> OddsQuote | None:
    snapshot = (
        session.query(OddsSnapshot)
        .filter(
            OddsSnapshot.game_id == game_id,
            OddsSnapshot.book == book,
            OddsSnapshot.market_type == market_type,
        )
        .order_by(OddsSnapshot.captured_at.desc())
        .first()
    )
    if snapshot is None:
        return None
    return OddsQuote(
        odds_home=snapshot.odds_home,
        odds_away=snapshot.odds_away,
        line=snapshot.line,
        captured_at=snapshot.captured_at,
        book=book,
    )


def _pending_bets_for_game(
    session: Session,
    *,
    game_id: str,
    book: str,
    market_type: str,
) -> list[PaperBet]:
    return (
        session.query(PaperBet)
        .filter(
            PaperBet.game_id == game_id,
            PaperBet.book == book,
            PaperBet.market_type == market_type,
            (PaperBet.outcome.is_(None)) | (PaperBet.outcome == "pending"),
        )
        .all()
    )


def _existing_pending(
    session: Session,
    *,
    game_id: str,
    book: str,
    side: str,
    market_type: str,
) -> PaperBet | None:
    return (
        session.query(PaperBet)
        .filter(
            PaperBet.game_id == game_id,
            PaperBet.book == book,
            PaperBet.side == side,
            PaperBet.market_type == market_type,
            (PaperBet.outcome.is_(None)) | (PaperBet.outcome == "pending"),
        )
        .first()
    )


def _estimate_edge_pct(
    session: Session,
    game: Game,
    settings: Settings,
    *,
    side: str,
    market_type: str,
    odds_taken: int,
    line_taken: float | None,
) -> float:
    loader = FeatureDataLoader(session)
    ctx = FeatureContext.for_game(game)
    pinnacle = loader.latest_odds_snapshot(
        ctx, book=settings.pinnacle_bookmaker, market_type=market_type
    )
    if pinnacle is None:
        return 0.0

    if market_type == "moneyline":
        home_odds = pinnacle.odds_home
        away_odds = pinnacle.odds_away
        if home_odds is None or away_odds is None:
            return 0.0
        fair_home, fair_away = devig_two_way(home_odds, away_odds)
        true_prob = fair_home if side == "home" else fair_away
        return expected_value(true_prob, odds_taken).edge_pct

    if line_taken is None or pinnacle.line is None:
        return 0.0
    # Spread edge vs sharp line is approximated from line difference for display only.
    if side == "home":
        line_edge = float(pinnacle.line) - float(line_taken)
        ref_odds = pinnacle.odds_home or -110
    else:
        line_edge = float(line_taken) - float(-pinnacle.line)
        ref_odds = pinnacle.odds_away or -110
    implied = american_to_implied_prob(ref_odds)
    return max(line_edge * implied * 10.0, 0.0)


class ManualBetPlacer:
    """Log a user-initiated paper bet (fake money today; real-money hook later)."""

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    def place(self, request: ManualBetRequest, *, replace_existing: bool = False) -> PlacementResult:
        game = self.session.query(Game).filter_by(game_id=request.game_id).one_or_none()
        if game is None:
            return PlacementResult(False, f"Game not found: {request.game_id}")

        side = request.side.lower().strip()
        if side not in {"home", "away"}:
            return PlacementResult(False, "Side must be home or away.")

        existing = _existing_pending(
            self.session,
            game_id=request.game_id,
            book=request.book,
            side=side,
            market_type=request.market_type,
        )
        if existing is not None:
            if replace_existing:
                return self._update_existing(existing, request, side=side)
            return PlacementResult(
                False,
                "You already have a pending bet on this game, book, side, and market.",
                duplicate=True,
                bet_id=existing.id,
            )

        if replace_existing:
            for stale in _pending_bets_for_game(
                self.session,
                game_id=request.game_id,
                book=request.book,
                market_type=request.market_type,
            ):
                if stale.side != side:
                    return self._update_existing(stale, request, side=side)

        stake_pct = request.stake_pct
        if stake_pct is None:
            stake_pct = min(self.settings.max_stake_pct, 1.0)
        if stake_pct <= 0:
            return PlacementResult(False, "Stake must be greater than 0% of bankroll.")

        edge_pct = request.edge_pct
        if edge_pct is None:
            edge_pct = _estimate_edge_pct(
                self.session,
                game,
                self.settings,
                side=side,
                market_type=request.market_type,
                odds_taken=request.odds_taken,
                line_taken=request.line_taken,
            )
        now = datetime.now(timezone.utc)
        audit = {
            "placement_type": "manual",
            "placed_at": now.isoformat(),
            "book": request.book,
            "market_type": request.market_type,
            "side": side,
            "american_odds": request.odds_taken,
            "line": request.line_taken,
            "stake_pct": stake_pct,
            "estimated_edge_pct": edge_pct,
            "notes": "User-initiated paper bet. Real-money execution is not wired yet.",
        }
        if request.audit_extra:
            audit.update(request.audit_extra)

        bet = PaperBet(
            game_id=request.game_id,
            book=request.book,
            market_type=request.market_type,
            side=side,
            flagged_at=now,
            suggested_stake_pct=stake_pct,
            odds_taken=request.odds_taken,
            line_taken=request.line_taken,
            model_version=request.model_version or "manual_v1",
            edge_pct=edge_pct,
            audit_trail=audit,
            outcome="pending",
        )
        self.session.add(bet)
        self.session.flush()
        return PlacementResult(True, "Paper bet placed.", bet_id=bet.id)

    def _update_existing(
        self,
        bet: PaperBet,
        request: ManualBetRequest,
        *,
        side: str,
    ) -> PlacementResult:
        stake_pct = request.stake_pct
        if stake_pct is None:
            stake_pct = min(self.settings.max_stake_pct, 1.0)
        if stake_pct <= 0:
            return PlacementResult(False, "Stake must be greater than 0% of bankroll.")

        edge_pct = request.edge_pct
        if edge_pct is None:
            game = self.session.query(Game).filter_by(game_id=request.game_id).one()
            edge_pct = _estimate_edge_pct(
                self.session,
                game,
                self.settings,
                side=side,
                market_type=request.market_type,
                odds_taken=request.odds_taken,
                line_taken=request.line_taken,
            )

        now = datetime.now(timezone.utc)
        audit = dict(bet.audit_trail or {})
        audit.update(
            {
                "placement_type": request.audit_extra.get("placement_type", "model_pick")
                if request.audit_extra
                else "manual",
                "updated_at": now.isoformat(),
                "book": request.book,
                "market_type": request.market_type,
                "side": side,
                "american_odds": request.odds_taken,
                "line": request.line_taken,
                "stake_pct": stake_pct,
                "estimated_edge_pct": edge_pct,
            }
        )
        if request.audit_extra:
            audit.update(request.audit_extra)

        bet.side = side
        bet.flagged_at = now
        bet.suggested_stake_pct = stake_pct
        bet.odds_taken = request.odds_taken
        bet.line_taken = request.line_taken
        bet.model_version = request.model_version or bet.model_version
        bet.edge_pct = edge_pct
        bet.audit_trail = audit
        bet.outcome = "pending"
        self.session.flush()
        return PlacementResult(
            True,
            "Paper bet updated.",
            bet_id=bet.id,
            updated=True,
        )
