from __future__ import annotations

from sports_ev.backtest.models import BetOutcome, BetSide, SettledBet
from sports_ev.pricing.odds import profit_if_win


def home_covered(*, home_score: int, away_score: int, home_line: float) -> BetOutcome:
    """
    Determine spread result for a home-side line.
    Home covers when: home_score + line > away_score (line is e.g. -3.5).
    """
    adjusted = home_score + home_line
    if adjusted > away_score:
        return BetOutcome.WIN
    if adjusted < away_score:
        return BetOutcome.LOSS
    return BetOutcome.PUSH


def away_covered(*, home_score: int, away_score: int, home_line: float) -> BetOutcome:
    """Away spread is inverse of home line."""
    home_result = home_covered(home_score=home_score, away_score=away_score, home_line=home_line)
    if home_result == BetOutcome.WIN:
        return BetOutcome.LOSS
    if home_result == BetOutcome.LOSS:
        return BetOutcome.WIN
    return BetOutcome.PUSH


def moneyline_outcome(*, side: str, home_score: int, away_score: int) -> BetOutcome:
    if home_score == away_score:
        return BetOutcome.PUSH
    home_won = home_score > away_score
    if side == BetSide.HOME.value or side == "home":
        return BetOutcome.WIN if home_won else BetOutcome.LOSS
    return BetOutcome.WIN if not home_won else BetOutcome.LOSS


def settle_spread_bet(
    *,
    game_id: str,
    book: str,
    side: BetSide,
    american_odds: int,
    line_taken: float,
    closing_line: float | None,
    true_prob: float,
    edge_pct: float,
    stake: float,
    home_score: int,
    away_score: int,
    kickoff_time,
) -> SettledBet:
    if side == BetSide.HOME:
        outcome = home_covered(home_score=home_score, away_score=away_score, home_line=line_taken)
    else:
        outcome = away_covered(home_score=home_score, away_score=away_score, home_line=line_taken)

    if outcome == BetOutcome.WIN:
        profit = profit_if_win(stake, american_odds)
    elif outcome == BetOutcome.LOSS:
        profit = -stake
    else:
        profit = 0.0

    clv = None
    if closing_line is not None:
        # Positive CLV means you beat the closing number on your side.
        if side == BetSide.HOME:
            clv = line_taken - closing_line
        else:
            clv = closing_line - line_taken

    return SettledBet(
        game_id=game_id,
        book=book,
        side=side,
        american_odds=american_odds,
        line_taken=line_taken,
        closing_line=closing_line,
        true_prob=true_prob,
        edge_pct=edge_pct,
        stake=stake,
        profit=profit,
        outcome=outcome,
        clv=clv,
        kickoff_time=kickoff_time,
        market_type="spread",
    )


def settle_moneyline_bet(
    *,
    game_id: str,
    book: str,
    side: BetSide,
    american_odds: int,
    closing_odds: int | None,
    true_prob: float,
    edge_pct: float,
    stake: float,
    home_score: int,
    away_score: int,
    kickoff_time,
) -> SettledBet:
    side_key = side.value if hasattr(side, "value") else str(side)
    outcome = moneyline_outcome(side=side_key, home_score=home_score, away_score=away_score)

    if outcome == BetOutcome.WIN:
        profit = profit_if_win(stake, american_odds)
    elif outcome == BetOutcome.LOSS:
        profit = -stake
    else:
        profit = 0.0

    clv = None
    if closing_odds is not None:
        from sports_ev.pricing.odds import american_to_implied_prob

        taken_implied = american_to_implied_prob(american_odds)
        closing_implied = american_to_implied_prob(closing_odds)
        clv = (closing_implied - taken_implied) * 100.0

    return SettledBet(
        game_id=game_id,
        book=book,
        side=side,
        american_odds=american_odds,
        line_taken=None,
        closing_line=None,
        true_prob=true_prob,
        edge_pct=edge_pct,
        stake=stake,
        profit=profit,
        outcome=outcome,
        clv=clv,
        kickoff_time=kickoff_time,
        market_type="moneyline",
    )
