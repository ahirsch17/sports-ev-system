"""Human-readable model scorecard from walk-forward backtest results."""

from __future__ import annotations

from sports_ev.backtest.models import BacktestReport, EdgeBucketStats
from sports_ev.config import Settings


def format_scorecard(report: BacktestReport, *, settings: Settings | None = None) -> str:
    settings = settings or Settings()
    lines = [
        "",
        "=" * 56,
        "  MODEL SCORECARD",
        "=" * 56,
        f"  Strategy     : {report.strategy_name}",
        f"  Min edge     : {settings.min_edge_pct:.1f}%",
        f"  Total bets   : {report.total_bets}",
        f"  Record       : {report.wins}-{report.losses}-{report.pushes} (W-L-P)",
        f"  Win rate     : {report.win_rate:.1f}%",
        f"  ROI          : {report.roi:+.2f}%",
        f"  Total profit : {report.total_profit:+.2f} units (flat stake)",
        f"  Avg CLV      : {_fmt_clv(report.avg_clv)}",
        f"  Max drawdown : {report.max_drawdown:.2f} units",
        "",
        "  Edge buckets (flat stake)",
        "  " + "-" * 52,
        f"  {'Bucket':<8} {'Bets':>5} {'Win%':>7} {'ROI':>8} {'Avg CLV':>10}",
    ]
    for bucket in report.edge_buckets:
        lines.append(_format_bucket_row(bucket))
    lines.extend(
        [
            "",
            "  How to read this",
            "  - Positive ROI + positive CLV at 2.5%+ edge = model may be usable.",
            "  - If 2.5%+ bucket is negative, do not bet real money yet.",
            "  - Walk-forward retrains on past games only (no lookahead).",
            "",
            f"  Reproducibility: {report.reproducibility_hash()[:16]}...",
            "=" * 56,
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_clv(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _format_bucket_row(bucket: EdgeBucketStats) -> str:
    if bucket.bets == 0:
        return f"  {bucket.bucket:<8} {'—':>5} {'—':>7} {'—':>8} {'—':>10}"
    return (
        f"  {bucket.bucket:<8} {bucket.bets:>5} {bucket.win_rate:>6.1f}% "
        f"{bucket.roi:>+7.1f}% {_fmt_clv(bucket.avg_clv):>10}"
    )
