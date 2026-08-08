from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean

from sqlalchemy.orm import Session

from sports_ev.backtest.report import EDGE_BUCKETS
from sports_ev.db.models import PaperBet
from sports_ev.paper.kelly import expected_profit_pct


def _edge_bucket_label(edge_pct: float) -> str:
    for label, low, high in EDGE_BUCKETS:
        if low <= edge_pct < high:
            return label
    return EDGE_BUCKETS[-1][0]


@dataclass(frozen=True)
class PaperBetSummary:
    total_bets: int
    pending: int
    settled: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi_pct: float
    avg_clv: float | None
    total_stake_pct: float
    clv_trend_positive: bool | None


@dataclass(frozen=True)
class ClvTrendPoint:
    settled_at: datetime
    bet_id: int
    clv: float
    cumulative_avg_clv: float


def _is_settled(bet: PaperBet) -> bool:
    return bet.outcome not in (None, "pending")


@dataclass(frozen=True)
class SportPerformanceRow:
    sport: str
    total_bets: int
    pending: int
    settled: int
    wins: int
    losses: int
    win_rate: float
    roi_pct: float
    avg_clv: float | None
    settled_pl_dollars: float


def compute_sport_breakdown(
    bets: list[PaperBet],
    *,
    sport_by_game_id: dict[str, str],
    money_views: dict[int, object] | None = None,
) -> list[SportPerformanceRow]:
    """Aggregate paper bet stats grouped by sport."""
    grouped: dict[str, list[PaperBet]] = {}
    for bet in bets:
        sport = sport_by_game_id.get(bet.game_id, "unknown")
        grouped.setdefault(sport, []).append(bet)

    rows: list[SportPerformanceRow] = []
    for sport in sorted(grouped):
        sport_bets = grouped[sport]
        summary = compute_summary(sport_bets)
        settled_pl = 0.0
        if money_views:
            for bet in sport_bets:
                if not _is_settled(bet):
                    continue
                view = money_views.get(bet.id)
                pl = getattr(view, "profit_loss_dollars", None)
                if pl is not None:
                    settled_pl += pl
        rows.append(
            SportPerformanceRow(
                sport=sport,
                total_bets=summary.total_bets,
                pending=summary.pending,
                settled=summary.settled,
                wins=summary.wins,
                losses=summary.losses,
                win_rate=summary.win_rate,
                roi_pct=summary.roi_pct,
                avg_clv=summary.avg_clv,
                settled_pl_dollars=settled_pl,
            )
        )
    return rows


@dataclass(frozen=True)
class PaperEdgeBucketRow:
    bucket: str
    bets: int
    wins: int
    losses: int
    win_rate: float
    roi_pct: float
    settled_pl_dollars: float
    avg_clv: float | None
    plus_ev_only: bool


def compute_paper_edge_buckets(
    bets: list[PaperBet],
    *,
    money_views: dict[int, object] | None = None,
    min_edge_pct: float = 2.5,
) -> list[PaperEdgeBucketRow]:
    """Group settled paper bets by edge bucket for performance review."""
    settled = [b for b in bets if _is_settled(b)]
    bucket_map: dict[str, list[PaperBet]] = {label: [] for label, _, _ in EDGE_BUCKETS}
    for bet in settled:
        bucket_map[_edge_bucket_label(bet.edge_pct)].append(bet)

    rows: list[PaperEdgeBucketRow] = []
    for label, _, _ in EDGE_BUCKETS:
        group = bucket_map[label]
        if not group:
            rows.append(
                PaperEdgeBucketRow(
                    bucket=label,
                    bets=0,
                    wins=0,
                    losses=0,
                    win_rate=0.0,
                    roi_pct=0.0,
                    settled_pl_dollars=0.0,
                    avg_clv=None,
                    plus_ev_only=_bucket_meets_min_edge(label, min_edge_pct),
                )
            )
            continue
        summary = compute_summary(group)
        settled_pl = 0.0
        if money_views:
            for bet in group:
                view = money_views.get(bet.id)
                pl = getattr(view, "profit_loss_dollars", None)
                if pl is not None:
                    settled_pl += pl
        rows.append(
            PaperEdgeBucketRow(
                bucket=label,
                bets=summary.settled,
                wins=summary.wins,
                losses=summary.losses,
                win_rate=summary.win_rate,
                roi_pct=summary.roi_pct,
                settled_pl_dollars=settled_pl,
                avg_clv=summary.avg_clv,
                plus_ev_only=_bucket_meets_min_edge(label, min_edge_pct),
            )
        )
    return rows


def _bucket_meets_min_edge(label: str, min_edge_pct: float) -> bool:
    for bucket_label, low, _high in EDGE_BUCKETS:
        if bucket_label == label:
            return low >= min_edge_pct
    return False


def compute_summary(bets: list[PaperBet]) -> PaperBetSummary:
    total = len(bets)
    pending = sum(1 for b in bets if not _is_settled(b))
    settled_bets = [b for b in bets if _is_settled(b)]
    settled = len(settled_bets)

    wins = sum(1 for b in settled_bets if b.outcome == "win")
    losses = sum(1 for b in settled_bets if b.outcome == "loss")
    pushes = sum(1 for b in settled_bets if b.outcome == "push")

    win_rate = (wins / settled * 100.0) if settled else 0.0

    profit_pcts: list[float] = []
    total_stake_pct = 0.0
    for bet in settled_bets:
        stake = bet.suggested_stake_pct
        total_stake_pct += stake
        won = bet.outcome == "win"
        pushed = bet.outcome == "push"
        profit_pcts.append(expected_profit_pct(stake, bet.odds_taken, won, pushed))

    roi_pct = (sum(profit_pcts) / total_stake_pct * 100.0) if total_stake_pct else 0.0

    clv_values = [b.clv for b in settled_bets if b.clv is not None]
    avg_clv = mean(clv_values) if clv_values else None

    trend = clv_trend(settled_bets)
    clv_trend_positive = None
    if len(trend) >= 2:
        clv_trend_positive = trend[-1].cumulative_avg_clv > trend[0].cumulative_avg_clv

    return PaperBetSummary(
        total_bets=total,
        pending=pending,
        settled=settled,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        roi_pct=roi_pct,
        avg_clv=avg_clv,
        total_stake_pct=total_stake_pct,
        clv_trend_positive=clv_trend_positive,
    )


def clv_trend(bets: list[PaperBet]) -> list[ClvTrendPoint]:
    settled = [b for b in bets if _is_settled(b) and b.clv is not None and b.settled_at is not None]
    settled.sort(key=lambda b: b.settled_at)

    points: list[ClvTrendPoint] = []
    running: list[float] = []
    for bet in settled:
        running.append(float(bet.clv))
        points.append(
            ClvTrendPoint(
                settled_at=bet.settled_at,
                bet_id=bet.id,
                clv=float(bet.clv),
                cumulative_avg_clv=mean(running),
            )
        )
    return points


def load_all_paper_bets(session: Session) -> list[PaperBet]:
    return session.query(PaperBet).order_by(PaperBet.flagged_at.desc()).all()


def load_paper_bets(session: Session, *, include_pending: bool = True) -> list[PaperBet]:
    query = session.query(PaperBet)
    if not include_pending:
        query = query.filter(PaperBet.outcome.notin_([None, "pending"]))
    return query.order_by(PaperBet.flagged_at.desc()).all()
