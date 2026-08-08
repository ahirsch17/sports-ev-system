"""Dashboard action wrappers. Same logic as CLI, callable from Streamlit buttons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sports_ev.config import Settings, get_settings
from sports_ev.db import session_scope
from sports_ev.db.models import DataSource, Game, SourceStatus
from sports_ev.fixtures.sync import PinnacleFixtureSync
from sports_ev.paper.flagging import OpportunityFlagger
from sports_ev.paper.placement import ManualBetPlacer, ManualBetRequest, latest_odds_quote
from sports_ev.paper.recommendations import get_model_picks, pick_is_bettable
from sports_ev.paper.settlement import PaperBetSettlementService
from sports_ev.scores.ingest import ScoreIngestService
from sports_ev.softbooks.ingest import SoftBookIngestService
from sports_ev.sports.registry import get_sport_config, list_sports


@dataclass
class ActionResult:
    success: bool
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemStatus:
    database_ok: bool
    sport: str
    pinnacle_scraper: str
    degraded_sources: list[str]
    upcoming_games: int
    games_tonight: int
    message: str


@dataclass
class OddsCoverage:
    upcoming_games: int
    pinnacle_games: int
    soft_book_games: int
    ev_ready_games: int
    soft_book: str
    market_type: str


def get_default_sport(
    *,
    settings: Settings | None = None,
    pending_bets: list | None = None,
    session=None,
) -> str:
    """Prefer sport with open paper bets, then nearest upcoming kickoff, else MLB."""
    settings = settings or get_settings()
    try:
        from collections import Counter
        from datetime import datetime, timezone

        from sports_ev.db.models import Game

        now = datetime.now(timezone.utc)

        def pick(db) -> str | None:
            if pending_bets:
                sport_counts: Counter[str] = Counter()
                for bet in pending_bets:
                    if (getattr(bet, "outcome", None) or "pending") != "pending":
                        continue
                    game = db.query(Game).filter_by(game_id=bet.game_id).one_or_none()
                    if game is not None:
                        sport_counts[game.sport] += 1
                if sport_counts:
                    return sport_counts.most_common(1)[0][0]

            nearest = (
                db.query(Game)
                .filter(Game.sport.in_(list_sports()), Game.kickoff_time > now)
                .order_by(Game.kickoff_time.asc())
                .first()
            )
            return nearest.sport if nearest is not None else None

        if session is not None:
            chosen = pick(session)
            if chosen:
                return chosen
        else:
            with session_scope(settings=settings) as scoped:
                chosen = pick(scoped)
                if chosen:
                    return chosen
    except Exception:
        pass
    return "mlb"


def get_odds_coverage(
    *,
    sport: str,
    soft_book: str = "draftkings",
    settings: Settings | None = None,
) -> OddsCoverage:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    market_type = config.default_market

    try:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        with session_scope(settings=settings) as session:
            games = (
                session.query(Game)
                .filter(Game.sport == sport, Game.kickoff_time > now)
                .order_by(Game.kickoff_time.asc())
                .all()
            )
            pinnacle_games = 0
            soft_book_games = 0
            ev_ready_games = 0
            for game in games:
                pinnacle = latest_odds_quote(
                    session,
                    game_id=game.game_id,
                    book=settings.pinnacle_bookmaker,
                    market_type=market_type,
                )
                soft = latest_odds_quote(
                    session,
                    game_id=game.game_id,
                    book=soft_book,
                    market_type=market_type,
                )
                if pinnacle is not None:
                    pinnacle_games += 1
                if soft is not None:
                    soft_book_games += 1
                if pinnacle is not None and soft is not None:
                    ev_ready_games += 1
            return OddsCoverage(
                upcoming_games=len(games),
                pinnacle_games=pinnacle_games,
                soft_book_games=soft_book_games,
                ev_ready_games=ev_ready_games,
                soft_book=soft_book,
                market_type=market_type,
            )
    except Exception:
        return OddsCoverage(0, 0, 0, 0, soft_book, config.default_market)


def get_system_status(*, sport: str = "nfl", settings: Settings | None = None) -> SystemStatus:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    degraded: list[str] = []
    upcoming = 0
    tonight = 0
    db_ok = False

    try:
        from datetime import datetime, timedelta, timezone

        with session_scope(settings=settings) as session:
            db_ok = True
            degraded = [
                s.source_key
                for s in session.query(DataSource).filter(DataSource.status == SourceStatus.DEGRADED).all()
            ]
            now = datetime.now(timezone.utc)
            tonight_end = now + timedelta(hours=24)

            upcoming = (
                session.query(Game)
                .filter(Game.sport == config.sport, Game.kickoff_time > now)
                .count()
            )
            tonight = (
                session.query(Game)
                .filter(
                    Game.sport == config.sport,
                    Game.kickoff_time > now,
                    Game.kickoff_time <= tonight_end,
                )
                .count()
            )
    except Exception as exc:
        return SystemStatus(
            database_ok=False,
            sport=config.sport,
            pinnacle_scraper="unavailable",
            degraded_sources=[],
            upcoming_games=0,
            games_tonight=0,
            message=f"Database error: {exc}",
        )

    parts = []
    if not db_ok:
        parts.append("Database not connected")
    if degraded:
        parts.append(f"{len(degraded)} degraded source(s)")
    if upcoming == 0:
        parts.append(f"No upcoming {config.display_name} games. Click Sync fixtures")
    else:
        coverage = get_odds_coverage(sport=sport, settings=settings)
        if coverage.pinnacle_games == 0:
            parts.append("Sharp lines missing. Click Poll Pinnacle")
        elif coverage.soft_book_games == 0:
            parts.append(
                f"Soft-book lines missing for {coverage.upcoming_games} upcoming game(s). "
                "Click Poll soft books to enable auto +EV flagging"
            )
        elif coverage.ev_ready_games < coverage.upcoming_games:
            parts.append(
                f"{coverage.ev_ready_games}/{coverage.upcoming_games} games have both sharp and soft lines"
            )

    return SystemStatus(
        database_ok=db_ok,
        sport=config.sport,
        pinnacle_scraper="guest API (free, scraped)",
        degraded_sources=degraded,
        upcoming_games=upcoming,
        games_tonight=tonight,
        message="; ".join(parts) if parts else "Ready",
    )


def sync_fixtures(*, sport: str = "nfl", settings: Settings | None = None) -> ActionResult:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    try:
        with session_scope(settings=settings) as session:
            syncer = PinnacleFixtureSync(session, settings)
            try:
                result = syncer.sync_fixtures(config.sport)
            finally:
                syncer.close()
    except Exception as exc:
        return ActionResult(False, f"Sync {config.display_name} Fixtures", str(exc))

    ok = result.games_upserted > 0 or result.matchups_seen > 0
    return ActionResult(
        ok,
        f"Sync {config.display_name} Fixtures",
        f"Upserted {result.games_upserted} games from {result.matchups_seen} Pinnacle matchups",
        {
            "sport": config.sport,
            "matchups_seen": result.matchups_seen,
            "games_upserted": result.games_upserted,
            "errors": result.errors,
        },
    )


def sync_all_fixtures(settings: Settings | None = None) -> list[ActionResult]:
    return [sync_fixtures(sport=sport, settings=settings) for sport in list_sports()]


def poll_pinnacle(*, sport: str = "nfl", settings: Settings | None = None) -> ActionResult:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    try:
        with session_scope(settings=settings) as session:
            svc = SoftBookIngestService(session, settings)
            result = svc.poll_books(["pinnacle"], sport=config.sport)
    except Exception as exc:
        return ActionResult(False, "Poll Pinnacle", str(exc))

    return ActionResult(
        result.snapshots_accepted > 0 or result.lines_scraped > 0,
        "Poll Pinnacle",
        f"Accepted {result.snapshots_accepted} Pinnacle snapshots ({result.snapshots_rejected} rejected)",
        {
            "sport": config.sport,
            "market_type": config.default_market,
            "lines_scraped": result.lines_scraped,
            "snapshots_accepted": result.snapshots_accepted,
            "snapshots_rejected": result.snapshots_rejected,
            "unmatched_games": result.unmatched_games,
            "errors": result.errors,
        },
    )


def poll_softbooks(
    *,
    sport: str = "nfl",
    books: list[str] | None = None,
    settings: Settings | None = None,
) -> ActionResult:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    books = books or ["draftkings", "fanduel"]
    try:
        with session_scope(settings=settings) as session:
            svc = SoftBookIngestService(session, settings)
            result = svc.poll_books(books, sport=config.sport)
    except Exception as exc:
        return ActionResult(False, "Poll Soft Books", str(exc))

    return ActionResult(
        result.snapshots_accepted > 0,
        "Poll Soft Books",
        f"Scraped {result.lines_scraped} lines, accepted {result.snapshots_accepted} snapshots",
        {
            "sport": config.sport,
            "market_type": config.default_market,
            "books_polled": result.books_polled,
            "lines_scraped": result.lines_scraped,
            "snapshots_accepted": result.snapshots_accepted,
            "snapshots_rejected": result.snapshots_rejected,
            "unmatched_games": result.unmatched_games,
            "errors": result.errors,
        },
    )


def sync_scores(
    *,
    sport: str | None = None,
    lookback_days: int = 3,
    settings: Settings | None = None,
) -> ActionResult:
    settings = settings or get_settings()
    try:
        with session_scope(settings=settings) as session:
            service = ScoreIngestService(session, settings)
            try:
                if sport:
                    result = service.ingest_sport(sport, lookback_days=lookback_days)
                    payload = {sport: result}
                else:
                    payload = service.ingest_all(lookback_days=lookback_days)
            finally:
                service.close()
    except Exception as exc:
        return ActionResult(False, "Sync Scores", str(exc))

    total_updated = sum(r.games_updated for r in payload.values())
    total_seen = sum(r.scores_seen for r in payload.values())
    errors: list[str] = []
    for key, res in payload.items():
        errors.extend(f"{key}: {err}" for err in res.errors)

    return ActionResult(
        total_updated > 0 or total_seen > 0,
        "Sync Scores",
        f"Updated {total_updated} game(s) from {total_seen} completed ESPN score(s)",
        {
            "sports": list(payload.keys()),
            "games_updated": total_updated,
            "scores_seen": total_seen,
            "errors": errors,
        },
    )


def flag_opportunities(
    *,
    sport: str = "nfl",
    days: int = 7,
    strategy: str = "market",
    soft_book: str = "draftkings",
    settings: Settings | None = None,
) -> ActionResult:
    settings = settings or get_settings()
    try:
        with session_scope(settings=settings) as session:
            result = OpportunityFlagger(session, settings).flag_upcoming(
                sport=sport,
                days_ahead=days,
                strategy=strategy,
                soft_book=soft_book,
            )
    except Exception as exc:
        return ActionResult(False, "Flag Opportunities", str(exc))

    return ActionResult(
        result.bets_flagged > 0 or result.games_scanned > 0,
        "Flag Opportunities",
        f"Flagged {result.bets_flagged} bet(s) from {result.games_scanned} game(s)",
        {
            "sport": result.sport,
            "market_type": result.market_type,
            "games_scanned": result.games_scanned,
            "opportunities_found": result.opportunities_found,
            "bets_flagged": result.bets_flagged,
            "skipped_existing": result.skipped_existing,
            "suppressed_degraded": result.suppressed_degraded,
            "errors": result.errors,
        },
    )


def log_model_picks(
    *,
    sport: str,
    soft_book: str = "draftkings",
    days: int = 7,
    only_plus_ev: bool = False,
    settings: Settings | None = None,
) -> ActionResult:
    settings = settings or get_settings()
    config = get_sport_config(sport, settings)
    try:
        with session_scope(settings=settings) as session:
            picks, errors = get_model_picks(
                session,
                sport=sport,
                soft_book=soft_book,
                days_ahead=days,
                settings=settings,
            )
            placer = ManualBetPlacer(session, settings)
            logged = 0
            updated = 0
            skipped = 0
            for pick in picks:
                if only_plus_ev and not pick_is_bettable(pick, settings):
                    skipped += 1
                    continue
                stake_pct = pick.suggested_stake_pct
                if stake_pct <= 0:
                    skipped += 1
                    continue
                result = placer.place(
                    ManualBetRequest(
                        game_id=pick.game_id,
                        book=soft_book,
                        market_type=config.default_market,
                        side=pick.side,
                        odds_taken=pick.american_odds,
                        line_taken=pick.line_taken,
                        stake_pct=stake_pct,
                        model_version=pick.model_version,
                        edge_pct=pick.edge_pct,
                        audit_extra={
                            "placement_type": "model_pick",
                            "model_prob": pick.model_prob,
                            "market_implied_prob": pick.market_implied_prob,
                            "matchup": pick.matchup,
                        },
                    ),
                    replace_existing=True,
                )
                if result.success and result.updated:
                    updated += 1
                elif result.success:
                    logged += 1
                elif result.duplicate:
                    skipped += 1
    except Exception as exc:
        return ActionResult(False, "Log Model Picks", str(exc))

    total = logged + updated
    if total > 0:
        message = f"Logged {logged} new pick(s) and updated {updated} existing pick(s)."
        if skipped:
            message += f" Skipped {skipped}."
    elif skipped:
        message = (
            f"No picks logged. Skipped {skipped}. "
            "Existing pending bets may already match these picks, or they did not meet the +EV filter."
        )
    else:
        message = "No model picks were available to log."

    return ActionResult(
        total > 0,
        "Log Model Picks",
        message,
        {"logged": logged, "updated": updated, "skipped": skipped, "errors": errors},
    )


def place_manual_bet(
    *,
    game_id: str,
    book: str,
    market_type: str,
    side: str,
    odds_taken: int,
    line_taken: float | None = None,
    stake_pct: float | None = None,
    settings: Settings | None = None,
) -> ActionResult:
    settings = settings or get_settings()
    try:
        with session_scope(settings=settings) as session:
            result = ManualBetPlacer(session, settings).place(
                ManualBetRequest(
                    game_id=game_id,
                    book=book,
                    market_type=market_type,
                    side=side,
                    odds_taken=odds_taken,
                    line_taken=line_taken,
                    stake_pct=stake_pct,
                )
            )
            if not result.success:
                return ActionResult(False, "Place Paper Bet", result.message, {"duplicate": result.duplicate})
    except Exception as exc:
        return ActionResult(False, "Place Paper Bet", str(exc))

    return ActionResult(
        True,
        "Place Paper Bet",
        result.message,
        {"bet_id": result.bet_id},
    )


def settle_paper_bets(settings: Settings | None = None) -> ActionResult:
    settings = settings or get_settings()
    try:
        with session_scope(settings=settings) as session:
            capture_closing_lines(session=session, settings=settings)
            result = PaperBetSettlementService(session, settings).settle_pending()
    except Exception as exc:
        return ActionResult(False, "Settle Paper Bets", str(exc))

    return ActionResult(
        True,
        "Settle Paper Bets",
        f"Settled {result.settled} bet(s), {result.still_pending} still pending",
        {
            "pending_checked": result.pending_checked,
            "settled": result.settled,
            "still_pending": result.still_pending,
            "errors": result.errors,
        },
    )


def capture_closing_lines(
    *,
    settings: Settings | None = None,
    session=None,
    lookback_hours: int = 48,
) -> ActionResult:
    settings = settings or get_settings()
    try:
        from sports_ev.odds.closing_capture import capture_closing_lines_for_all_sports

        if session is not None:
            result = capture_closing_lines_for_all_sports(
                session, settings=settings, lookback_hours=lookback_hours
            )
        else:
            with session_scope(settings=settings) as scoped:
                result = capture_closing_lines_for_all_sports(
                    scoped, settings=settings, lookback_hours=lookback_hours
                )
    except Exception as exc:
        return ActionResult(False, "Capture Closing Lines", str(exc))

    return ActionResult(
        True,
        "Capture Closing Lines",
        f"Captured {result.games_captured} closing line(s) "
        f"({result.skipped_existing} already stored, {result.games_checked} checked)",
        {
            "games_checked": result.games_checked,
            "games_captured": result.games_captured,
            "skipped_existing": result.skipped_existing,
            "errors": result.errors,
        },
    )


def sync_mlb_probables(
    *,
    settings: Settings | None = None,
    days_ahead: int = 7,
) -> ActionResult:
    settings = settings or get_settings()
    try:
        from sports_ev.stats.mlb_probables import MlbProbableSyncService

        with session_scope(settings=settings) as session:
            result = MlbProbableSyncService(session).sync_upcoming(days_ahead=days_ahead)
    except Exception as exc:
        return ActionResult(False, "Sync MLB Probables", str(exc))

    return ActionResult(
        result.games_updated > 0 or result.games_checked > 0,
        "Sync MLB Probables",
        f"Updated {result.games_updated} game(s) from {result.games_checked} upcoming MLB game(s)",
        {
            "games_checked": result.games_checked,
            "games_updated": result.games_updated,
            "stats_inserted": result.stats_inserted,
            "errors": result.errors,
        },
    )


def run_full_refresh(
    *,
    sport: str = "nfl",
    days: int = 7,
    strategy: str = "market",
    soft_book: str = "draftkings",
    settings: Settings | None = None,
) -> list[ActionResult]:
    settings = settings or get_settings()
    return [
        sync_fixtures(sport=sport, settings=settings),
        poll_pinnacle(sport=sport, settings=settings),
        poll_softbooks(sport=sport, settings=settings),
        flag_opportunities(
            sport=sport,
            days=days,
            strategy=strategy,
            soft_book=soft_book,
            settings=settings,
        ),
    ]
