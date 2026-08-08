from __future__ import annotations

from dataclasses import dataclass

from sports_ev.pricing.odds import american_to_implied_prob, profit_if_win


@dataclass(frozen=True)
class EvResult:
    ev_dollars: float
    ev_pct: float
    edge_pct: float
    true_prob: float
    offered_implied_prob: float
    profit_if_win: float
    stake: float


def expected_value(
    true_prob: float,
    american_odds: int,
    stake: float = 1.0,
) -> EvResult:
    """
    EV = (true_prob × profit_if_win) − ((1 − true_prob) × stake)

    edge_pct is EV as a percentage of stake.
    """
    if not 0.0 <= true_prob <= 1.0:
        raise ValueError("true_prob must be between 0 and 1")
    if stake <= 0:
        raise ValueError("Stake must be positive")

    win_profit = profit_if_win(stake, american_odds)
    loss_prob = 1.0 - true_prob
    ev_dollars = true_prob * win_profit - loss_prob * stake
    offered_implied = american_to_implied_prob(american_odds)

    return EvResult(
        ev_dollars=ev_dollars,
        ev_pct=(ev_dollars / stake) * 100.0,
        edge_pct=(true_prob - offered_implied) * 100.0,
        true_prob=true_prob,
        offered_implied_prob=offered_implied,
        profit_if_win=win_profit,
        stake=stake,
    )


def is_plus_ev(
    true_prob: float,
    american_odds: int,
    *,
    min_edge_pct: float = 2.0,
    stake: float = 1.0,
) -> bool:
    """True when EV is positive and edge meets the minimum threshold."""
    result = expected_value(true_prob, american_odds, stake=stake)
    return result.ev_dollars > 0 and result.edge_pct >= min_edge_pct
