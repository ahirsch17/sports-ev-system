from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class BetSide(StrEnum):
    HOME = "home"
    AWAY = "away"


class BetOutcome(StrEnum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"


@dataclass(frozen=True)
class BetCandidate:
    game_id: str
    book: str
    side: BetSide
    american_odds: int
    line: float | None
    true_prob: float
    edge_pct: float
    decision_time: datetime
    stake: float = 1.0
    market_type: str = "spread"


@dataclass(frozen=True)
class SettledBet:
    game_id: str
    book: str
    side: BetSide
    american_odds: int
    line_taken: float | None
    closing_line: float | None
    true_prob: float
    edge_pct: float
    stake: float
    profit: float
    outcome: BetOutcome
    clv: float | None
    kickoff_time: datetime
    market_type: str = "spread"


@dataclass(frozen=True)
class EdgeBucketStats:
    bucket: str
    bets: int
    wins: int
    win_rate: float
    roi: float
    avg_clv: float | None


@dataclass
class BacktestReport:
    backtest_version: str
    strategy_name: str
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi: float
    total_profit: float
    avg_clv: float | None
    max_drawdown: float
    edge_buckets: list[EdgeBucketStats]
    settled_bets: list[SettledBet]

    def reproducibility_hash(self) -> str:
        import hashlib
        import json

        payload = {
            "backtest_version": self.backtest_version,
            "strategy_name": self.strategy_name,
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "roi": round(self.roi, 8),
            "max_drawdown": round(self.max_drawdown, 8),
            "bets": [
                {
                    "game_id": b.game_id,
                    "side": b.side,
                    "profit": round(b.profit, 8),
                    "outcome": b.outcome,
                    "edge_pct": round(b.edge_pct, 8),
                }
                for b in self.settled_bets
            ],
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
