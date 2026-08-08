"""Paper bankroll helpers: translate stake % into simulated dollar amounts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sports_ev.db.models import PaperBet
from sports_ev.paper.kelly import expected_profit_pct
from sports_ev.pricing.odds import profit_if_win


@dataclass(frozen=True)
class PaperBankrollSnapshot:
    starting_bankroll: float
    current_bankroll: float
    at_risk: float
    available: float
    settled_profit: float


@dataclass(frozen=True)
class BetMoneyView:
    stake_dollars: float
    to_win_dollars: float
    payout_dollars: float
    profit_loss_dollars: float | None


def _is_pending(bet: PaperBet) -> bool:
    return bet.outcome in (None, "pending")


def stake_dollars_from_pct(bankroll: float, stake_pct: float) -> float:
    return max(bankroll * (stake_pct / 100.0), 0.0)


def pick_money_preview(
    *,
    bankroll: float,
    stake_pct: float,
    american_odds: int,
) -> BetMoneyView:
    """Dollar preview for a model pick before it is logged."""
    stake = stake_dollars_from_pct(bankroll, stake_pct)
    to_win = profit_if_win(stake, american_odds) if stake > 0 else 0.0
    return BetMoneyView(
        stake_dollars=stake,
        to_win_dollars=to_win,
        payout_dollars=stake + to_win,
        profit_loss_dollars=None,
    )


def bet_money_view(bet: PaperBet, *, bankroll_at_bet: float) -> BetMoneyView:
    stake = stake_dollars_from_pct(bankroll_at_bet, bet.suggested_stake_pct)
    to_win = profit_if_win(stake, bet.odds_taken) if stake > 0 else 0.0
    payout = stake + to_win
    profit_loss = None
    if not _is_pending(bet):
        won = bet.outcome == "win"
        pushed = bet.outcome == "push"
        profit_pct = expected_profit_pct(
            bet.suggested_stake_pct,
            bet.odds_taken,
            won,
            pushed,
        )
        profit_loss = bankroll_at_bet * (profit_pct / 100.0)
    return BetMoneyView(
        stake_dollars=stake,
        to_win_dollars=to_win,
        payout_dollars=payout,
        profit_loss_dollars=profit_loss,
    )


def compute_bankroll_state(
    bets: list[PaperBet],
    *,
    starting_bankroll: float,
) -> tuple[PaperBankrollSnapshot, dict[int, BetMoneyView]]:
    """Walk bets in placement order to size stakes and update running bankroll."""
    ordered = sorted(
        bets,
        key=lambda b: b.flagged_at or datetime.min.replace(tzinfo=timezone.utc),
    )
    running = starting_bankroll
    at_risk = 0.0
    views: dict[int, BetMoneyView] = {}

    for bet in ordered:
        view = bet_money_view(bet, bankroll_at_bet=running)
        views[bet.id] = view
        if _is_pending(bet):
            at_risk += view.stake_dollars
        else:
            running += view.profit_loss_dollars or 0.0

    settled_profit = running - starting_bankroll
    return (
        PaperBankrollSnapshot(
            starting_bankroll=starting_bankroll,
            current_bankroll=running,
            at_risk=at_risk,
            available=max(running - at_risk, 0.0),
            settled_profit=settled_profit,
        ),
        views,
    )


def format_dollars(amount: float) -> str:
    return f"${amount:,.2f}"


def format_signed_dollars(amount: float) -> str:
    if amount >= 0:
        return f"+{format_dollars(amount)}"
    return f"-{format_dollars(abs(amount))}"
