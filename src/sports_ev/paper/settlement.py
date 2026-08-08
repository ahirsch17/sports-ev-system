from __future__ import annotations



from dataclasses import dataclass, field

from datetime import datetime, timezone



from sqlalchemy.orm import Session



from sports_ev.backtest.models import BetOutcome, BetSide

from sports_ev.backtest.settlement import away_covered, home_covered, moneyline_outcome

from sports_ev.config import Settings, get_settings

from sports_ev.db.models import Game, PaperBet

from sports_ev.features.sources import FeatureContext, FeatureDataLoader

from sports_ev.pricing.odds import american_to_implied_prob





@dataclass

class SettlementResult:

    pending_checked: int = 0

    settled: int = 0

    still_pending: int = 0

    errors: list[str] = field(default_factory=list)





def _closing_line(session: Session, game: Game, settings: Settings, book: str) -> float | None:
    if game.pinnacle_closing_spread is not None:
        return game.pinnacle_closing_spread

    loader = FeatureDataLoader(session)

    ctx = FeatureContext.for_game(game)

    pinnacle = loader.latest_odds_snapshot(

        ctx, book=settings.pinnacle_bookmaker, market_type="spread"

    )

    if pinnacle and pinnacle.line is not None:

        return pinnacle.line

    soft = loader.latest_odds_snapshot(ctx, book=book, market_type="spread")

    return soft.line if soft else None





def _closing_odds(
    session: Session, game: Game, settings: Settings, book: str, side: str, market_type: str
) -> int | None:
    if side == BetSide.HOME.value or side == "home":
        if game.pinnacle_closing_home_odds is not None:
            return int(game.pinnacle_closing_home_odds)
    elif game.pinnacle_closing_away_odds is not None:
        return int(game.pinnacle_closing_away_odds)

    loader = FeatureDataLoader(session)

    ctx = FeatureContext.for_game(game)

    pinnacle = loader.latest_odds_snapshot(

        ctx, book=settings.pinnacle_bookmaker, market_type=market_type

    )

    if pinnacle is None:

        return None

    if side == BetSide.HOME.value or side == "home":

        return pinnacle.odds_home

    return pinnacle.odds_away





def _compute_clv(side: str, line_taken: float, closing_line: float | None) -> float | None:

    if closing_line is None:

        return None

    if side == BetSide.HOME.value or side == "home":

        return line_taken - closing_line

    return closing_line - line_taken





def _compute_moneyline_clv(odds_taken: int, closing_odds: int | None) -> float | None:

    if closing_odds is None:

        return None

    taken_implied = american_to_implied_prob(odds_taken)

    closing_implied = american_to_implied_prob(closing_odds)

    return (closing_implied - taken_implied) * 100.0





def _spread_outcome(side: str, home_score: int, away_score: int, line_taken: float) -> BetOutcome:
    if side == BetSide.HOME.value or side == "home":
        return home_covered(home_score=home_score, away_score=away_score, home_line=line_taken)
    home_line = -float(line_taken)
    return away_covered(home_score=home_score, away_score=away_score, home_line=home_line)





class PaperBetSettlementService:

    """Backfill win/loss/push and CLV once games settle."""



    def __init__(self, session: Session, settings: Settings | None = None):

        self.session = session

        self.settings = settings or get_settings()



    def settle_pending(self) -> SettlementResult:

        result = SettlementResult()

        pending = (

            self.session.query(PaperBet)

            .filter((PaperBet.outcome.is_(None)) | (PaperBet.outcome == "pending"))

            .all()

        )

        result.pending_checked = len(pending)



        for bet in pending:

            game = self.session.query(Game).filter_by(game_id=bet.game_id).one_or_none()

            if game is None:

                result.errors.append(f"Game not found for paper bet {bet.id}")

                continue

            if game.home_score is None or game.away_score is None:

                result.still_pending += 1

                continue



            market_type = bet.market_type or "spread"

            if market_type == "moneyline":

                outcome = moneyline_outcome(

                    side=bet.side,

                    home_score=int(game.home_score),

                    away_score=int(game.away_score),

                )

                closing_odds = _closing_odds(

                    self.session, game, self.settings, bet.book, bet.side, market_type

                )

                clv = _compute_moneyline_clv(int(bet.odds_taken), closing_odds)

                bet.outcome = outcome.value

                bet.closing_line = float(closing_odds) if closing_odds is not None else None

                bet.clv = clv

            else:

                if bet.line_taken is None:

                    result.errors.append(f"Paper bet {bet.id} missing line_taken")

                    continue

                outcome = _spread_outcome(

                    bet.side,

                    int(game.home_score),

                    int(game.away_score),

                    float(bet.line_taken),

                )

                closing = _closing_line(self.session, game, self.settings, bet.book)

                clv = _compute_clv(bet.side, float(bet.line_taken), closing)

                bet.outcome = outcome.value

                bet.closing_line = closing

                bet.clv = clv



            bet.settled_at = datetime.now(timezone.utc)

            result.settled += 1



        return result

