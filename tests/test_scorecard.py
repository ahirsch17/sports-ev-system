from __future__ import annotations

from datetime import datetime, timezone

from sports_ev.backtest.models import BacktestReport, BetOutcome, EdgeBucketStats, SettledBet, BetSide
from sports_ev.backtest.report import BACKTEST_VERSION, build_report
from sports_ev.backtest.scorecard import format_scorecard
from sports_ev.config import Settings


def _sample_settled_bet(*, edge_pct: float, outcome: BetOutcome = BetOutcome.WIN) -> SettledBet:
    return SettledBet(
        game_id="g1",
        book="draftkings",
        side=BetSide.HOME,
        american_odds=-110,
        line_taken=-3.5,
        closing_line=-3.5,
        true_prob=0.55,
        edge_pct=edge_pct,
        stake=1.0,
        profit=(100 / 110) if outcome == BetOutcome.WIN else -1.0,
        outcome=outcome,
        clv=0.5,
        kickoff_time=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )


class TestFormatScorecard:
    def test_includes_header_and_summary(self):
        report = build_report(
            strategy_name="test_strategy",
            settled_bets=[_sample_settled_bet(edge_pct=5.0)],
        )
        text = format_scorecard(report, settings=Settings(min_edge_pct=2.5))

        assert "MODEL SCORECARD" in text
        assert "test_strategy" in text
        assert "Min edge     : 2.5%" in text
        assert "Total bets   : 1" in text
        assert "Record       : 1-0-0 (W-L-P)" in text
        assert "Edge buckets (flat stake)" in text
        assert "How to read this" in text

    def test_formats_empty_bucket_as_dashes(self):
        report = BacktestReport(
            backtest_version=BACKTEST_VERSION,
            strategy_name="empty_buckets",
            total_bets=0,
            wins=0,
            losses=0,
            pushes=0,
            win_rate=0.0,
            roi=0.0,
            total_profit=0.0,
            avg_clv=None,
            max_drawdown=0.0,
            edge_buckets=[
                EdgeBucketStats(bucket="0-2%", bets=0, wins=0, win_rate=0.0, roi=0.0, avg_clv=None),
                EdgeBucketStats(bucket="2-4%", bets=0, wins=0, win_rate=0.0, roi=0.0, avg_clv=None),
            ],
            settled_bets=[],
        )
        text = format_scorecard(report)

        empty_line = next(line for line in text.splitlines() if line.strip().startswith("0-2%"))
        assert empty_line.count("—") == 4

    def test_formats_populated_bucket_row(self):
        report = build_report(
            strategy_name="bucket_test",
            settled_bets=[
                _sample_settled_bet(edge_pct=5.0, outcome=BetOutcome.WIN),
                _sample_settled_bet(edge_pct=5.0, outcome=BetOutcome.LOSS),
            ],
        )
        text = format_scorecard(report)
        bucket_line = next(line for line in text.splitlines() if "4-6%" in line)

        assert "2" in bucket_line
        assert "50.0%" in bucket_line
        assert "+0.50%" in bucket_line

    def test_includes_reproducibility_hash_prefix(self):
        report = build_report(strategy_name="hash_test", settled_bets=[_sample_settled_bet(edge_pct=3.0)])
        text = format_scorecard(report)

        assert "Reproducibility:" in text
        assert report.reproducibility_hash()[:16] in text
