from __future__ import annotations

from statistics import mean

from sports_ev.backtest.models import BacktestReport, BetOutcome, EdgeBucketStats, SettledBet

BACKTEST_VERSION = "walk_forward_v1"

EDGE_BUCKETS = [
    ("0-2%", 0.0, 2.0),
    ("2-4%", 2.0, 4.0),
    ("4-6%", 4.0, 6.0),
    ("6%+", 6.0, float("inf")),
]


def _max_drawdown(profits: list[float]) -> float:
    if not profits:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in profits:
        cumulative += p
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def _edge_bucket(edge_pct: float) -> str:
    for label, low, high in EDGE_BUCKETS:
        if low <= edge_pct < high:
            return label
    return EDGE_BUCKETS[-1][0]


def build_report(
    *,
    strategy_name: str,
    settled_bets: list[SettledBet],
) -> BacktestReport:
    total = len(settled_bets)
    wins = sum(1 for b in settled_bets if b.outcome == BetOutcome.WIN)
    losses = sum(1 for b in settled_bets if b.outcome == BetOutcome.LOSS)
    pushes = sum(1 for b in settled_bets if b.outcome == BetOutcome.PUSH)

    total_staked = sum(b.stake for b in settled_bets)
    total_profit = sum(b.profit for b in settled_bets)
    roi = (total_profit / total_staked * 100.0) if total_staked else 0.0
    win_rate = (wins / total * 100.0) if total else 0.0

    clv_values = [b.clv for b in settled_bets if b.clv is not None]
    avg_clv = mean(clv_values) if clv_values else None

    max_drawdown = _max_drawdown([b.profit for b in settled_bets])

    bucket_map: dict[str, list[SettledBet]] = {label: [] for label, _, _ in EDGE_BUCKETS}
    for bet in settled_bets:
        bucket_map[_edge_bucket(bet.edge_pct)].append(bet)

    edge_buckets: list[EdgeBucketStats] = []
    for label, _, _ in EDGE_BUCKETS:
        bets = bucket_map[label]
        if not bets:
            edge_buckets.append(
                EdgeBucketStats(bucket=label, bets=0, wins=0, win_rate=0.0, roi=0.0, avg_clv=None)
            )
            continue
        b_wins = sum(1 for b in bets if b.outcome == BetOutcome.WIN)
        b_staked = sum(b.stake for b in bets)
        b_profit = sum(b.profit for b in bets)
        b_clv = [b.clv for b in bets if b.clv is not None]
        edge_buckets.append(
            EdgeBucketStats(
                bucket=label,
                bets=len(bets),
                wins=b_wins,
                win_rate=(b_wins / len(bets) * 100.0),
                roi=(b_profit / b_staked * 100.0) if b_staked else 0.0,
                avg_clv=mean(b_clv) if b_clv else None,
            )
        )

    return BacktestReport(
        backtest_version=BACKTEST_VERSION,
        strategy_name=strategy_name,
        total_bets=total,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        roi=roi,
        total_profit=total_profit,
        avg_clv=avg_clv,
        max_drawdown=max_drawdown,
        edge_buckets=edge_buckets,
        settled_bets=settled_bets,
    )
