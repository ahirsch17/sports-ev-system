"""Human-readable paper betting performance report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import Game, PaperBet
from sports_ev.paper.bankroll import compute_bankroll_state, format_signed_dollars
from sports_ev.paper.metrics import compute_paper_edge_buckets, compute_summary, load_all_paper_bets


@dataclass(frozen=True)
class PaperReport:
    generated_at: datetime
    window_days: int
    text: str


def _bet_in_window(bet: PaperBet, *, since: datetime) -> bool:
    flagged = bet.flagged_at
    if flagged and flagged.tzinfo is None:
        flagged = flagged.replace(tzinfo=timezone.utc)
    settled = bet.settled_at
    if settled and settled.tzinfo is None:
        settled = settled.replace(tzinfo=timezone.utc)
    if flagged and flagged >= since:
        return True
    if settled and settled >= since:
        return True
    return False


def build_paper_report(
    session: Session,
    *,
    days: int = 7,
    settings: Settings | None = None,
) -> PaperReport:
    settings = settings or get_settings()
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    all_bets = load_all_paper_bets(session)
    window_bets = [b for b in all_bets if _bet_in_window(b, since=since)]
    games = {g.game_id: g for g in session.query(Game).all()}

    all_summary = compute_summary(all_bets)
    window_summary = compute_summary(window_bets)
    plus_ev_all = [b for b in all_bets if b.edge_pct >= settings.min_edge_pct]
    plus_ev_window = [b for b in window_bets if b.edge_pct >= settings.min_edge_pct]
    plus_summary = compute_summary(plus_ev_all)
    plus_window = compute_summary(plus_ev_window)

    snap, money_views = compute_bankroll_state(all_bets, starting_bankroll=settings.paper_bankroll)
    buckets = compute_paper_edge_buckets(
        all_bets,
        money_views=money_views,
        min_edge_pct=settings.min_edge_pct,
    )

    lines: list[str] = [
        "SportsPredictor · Paper report",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Window: last {days} days (flagged or settled in window)",
        "",
        "── Bankroll (all time) ──",
        f"  Starting: ${settings.paper_bankroll:,.2f}",
        f"  Current:  ${snap.current_bankroll:,.2f}",
        f"  Settled P/L: {format_signed_dollars(snap.settled_profit)}",
        f"  At risk (pending): ${snap.at_risk:,.2f}",
        "",
        f"── Last {days} days ──",
        f"  Bets in window: {len(window_bets)}",
        f"  Settled: {window_summary.settled} ({window_summary.wins}W-{window_summary.losses}L)",
        f"  ROI (stake-weighted): {window_summary.roi_pct:+.2f}%",
        f"  Avg CLV: {window_summary.avg_clv:.2f}" if window_summary.avg_clv is not None else "  Avg CLV: n/a",
        "",
        f"── +EV only (edge >= {settings.min_edge_pct:.1f}%) ──",
        f"  All time: {plus_summary.settled} settled, {plus_summary.wins}W-{plus_summary.losses}L, ROI {plus_summary.roi_pct:+.2f}%",
        f"  Last {days}d: {plus_window.settled} settled, {plus_window.wins}W-{plus_window.losses}L, ROI {plus_window.roi_pct:+.2f}%",
        "",
        "── All time ──",
        f"  Total bets: {all_summary.total_bets} ({all_summary.pending} pending, {all_summary.settled} settled)",
        f"  Record: {all_summary.wins}W-{all_summary.losses}L-{all_summary.pushes}P",
        f"  ROI: {all_summary.roi_pct:+.2f}%",
        f"  Avg CLV: {all_summary.avg_clv:.2f}" if all_summary.avg_clv is not None else "  Avg CLV: n/a",
        "",
        "── Edge buckets (settled) ──",
    ]
    for row in buckets:
        if row.bets == 0:
            continue
        tag = " +EV" if row.plus_ev_only else ""
        lines.append(
            f"  {row.bucket}{tag}: {row.bets} bets, {row.win_rate:.0f}% win, "
            f"ROI {row.roi_pct:+.1f}%, P/L {format_signed_dollars(row.settled_pl_dollars)}"
        )

    pending = [b for b in all_bets if b.outcome in (None, "pending")]
    if pending:
        lines.extend(["", "── Pending ──"])
        for bet in sorted(pending, key=lambda b: b.flagged_at or now)[:15]:
            game = games.get(bet.game_id)
            if game:
                label = f"{game.away_team} @ {game.home_team} ({game.sport})"
                kick = game.kickoff_time.strftime("%Y-%m-%d") if game.kickoff_time else "?"
            else:
                label = bet.game_id
                kick = "?"
            lines.append(
                f"  {kick} {label} | {bet.side} {bet.market_type} @{bet.odds_taken} | edge {bet.edge_pct:+.1f}%"
            )
        if len(pending) > 15:
            lines.append(f"  ... and {len(pending) - 15} more")

    if window_bets:
        lines.extend(["", f"── Settled in last {days} days ──"])
        for bet in sorted(window_bets, key=lambda b: b.settled_at or b.flagged_at or now):
            if bet.outcome in (None, "pending"):
                continue
            game = games.get(bet.game_id)
            label = f"{game.away_team} @ {game.home_team}" if game else bet.game_id
            pl = money_views.get(bet.id)
            pl_str = format_signed_dollars(pl.profit_loss_dollars) if pl and pl.profit_loss_dollars is not None else "—"
            lines.append(
                f"  {bet.outcome.upper()} {label} | edge {bet.edge_pct:+.1f}% | CLV {bet.clv} | {pl_str}"
            )

    lines.append("")
    return PaperReport(generated_at=now, window_days=days, text="\n".join(lines))


def format_paper_report(session: Session, *, days: int = 7, settings: Settings | None = None) -> str:
    return build_paper_report(session, days=days, settings=settings).text
