"""Unified SportsPredictor dashboard: Streamlit control room backed by sports-ev-system."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from sports_ev.config import get_settings
from sports_ev.dashboard.heartbeat import HEARTBEAT_INTERVAL_SECONDS, heartbeat_listen_port
from sports_ev.dashboard.actions import (
    flag_opportunities,
    get_default_sport,
    get_odds_coverage,
    get_system_status,
    log_model_picks,
    place_manual_bet,
    poll_pinnacle,
    poll_softbooks,
    run_full_refresh,
    settle_paper_bets,
    sync_fixtures,
    sync_scores,
)
from sports_ev.dashboard.help_copy import (
    DAYS_AHEAD_HELP,
    EDGE_COLUMN_HELP,
    MODEL_PICKS_CAPTION,
    PAPER_BANKROLL_HELP,
    PENDING_METRIC_HELP,
    PAYOUT_DOLLARS_HELP,
    PROFIT_LOSS_HELP,
    REFRESH_ODDS_HELP,
    ROI_METRIC_HELP,
    SETTLE_FINISHED_GAMES_HELP,
    SOFT_BOOK_HELP,
    STAKE_DOLLARS_HELP,
    STRATEGY_HELP,
    TO_WIN_DOLLARS_HELP,
    TYPE_COLUMN_HELP,
    GAME_CLOCK_HELP,
    SPORT_PERFORMANCE_CAPTION,
    log_all_picks_help,
    log_plus_ev_only_help,
    sidebar_log_model_picks_help,
)
from sports_ev.dashboard.theme import APP_TITLE, THEME_CSS, hero_html
from sports_ev.db import session_scope
from sports_ev.db.models import Game, PaperBet
from sports_ev.paper.bankroll import (
    BetMoneyView,
    compute_bankroll_state,
    format_dollars,
    format_signed_dollars,
    pick_money_preview,
)
from sports_ev.paper.metrics import (
    clv_trend,
    compute_paper_edge_buckets,
    compute_sport_breakdown,
    compute_summary,
    load_all_paper_bets,
)
from sports_ev.paper.placement import latest_odds_quote
from sports_ev.paper.recommendations import (
    get_all_sports_picks_24h,
    get_model_picks,
    pick_is_bettable,
    sport_has_model,
)
from sports_ev.game_clock import (
    format_game_clock,
    game_clock_for,
    load_live_status_by_game_id,
)
from sports_ev.sports.registry import get_sport_config, list_sports

LIVE_STATUS_TTL_SECONDS = 120


def _render_browser_heartbeat() -> None:
    """Browser-only ping via localhost HTTP — runner shuts down when tab closes."""
    hb_port = heartbeat_listen_port(int(os.environ.get("SPORTS_EV_DASHBOARD_PORT", "8501")))
    interval_ms = HEARTBEAT_INTERVAL_SECONDS * 1000
    st.iframe(
        f"""
        <script>
        (function() {{
            const root = window.parent !== window ? window.parent : window;
            if (root.__sportsEvHeartbeat) return;
            root.__sportsEvHeartbeat = true;
            const pingUrl = "http://127.0.0.1:{hb_port}/ping";
            const byeUrl = "http://127.0.0.1:{hb_port}/bye";
            const focusUrl = "http://127.0.0.1:{hb_port}/focus";
            let lastFocusAt = 0;
            function ping() {{
                try {{
                    fetch(pingUrl + "?t=" + Date.now(), {{ mode: "cors", cache: "no-store" }});
                }} catch (e) {{
                    new Image().src = pingUrl + "?t=" + Date.now();
                }}
            }}
            function bye() {{
                try {{
                    if (navigator.sendBeacon) {{
                        navigator.sendBeacon(byeUrl);
                    }} else {{
                        fetch(byeUrl, {{ mode: "cors", keepalive: true }});
                    }}
                }} catch (e) {{}}
            }}
            function nudgeFocus() {{
                try {{ root.focus(); }} catch (e) {{}}
                try {{
                    const doc = root.document;
                    const prev = doc.title;
                    if (!prev.startsWith("→ ")) {{
                        doc.title = "→ " + prev;
                        setTimeout(() => {{
                            if (doc.title.startsWith("→ ")) doc.title = prev;
                        }}, 1600);
                    }}
                }} catch (e) {{}}
            }}
            function pollFocus() {{
                fetch(focusUrl + "?t=" + Date.now(), {{ mode: "cors", cache: "no-store" }})
                    .then((r) => r.json())
                    .then((data) => {{
                        const at = Number(data && data.at) || 0;
                        if (at > lastFocusAt) {{
                            if (lastFocusAt > 0) nudgeFocus();
                            lastFocusAt = at;
                        }} else if (!lastFocusAt && at) {{
                            lastFocusAt = at;
                        }}
                    }})
                    .catch(() => {{}});
            }}
            ping();
            pollFocus();
            setInterval(ping, {interval_ms});
            setInterval(pollFocus, 1200);
            root.addEventListener("pagehide", bye);
            root.addEventListener("beforeunload", bye);
        }})();
        </script>
        """,
        height=1,
        width=1,
    )


def _sport_label(sport_key: str) -> str:
    try:
        return get_sport_config(sport_key).display_name
    except KeyError:
        return sport_key.upper() if sport_key != "unknown" else "Unknown"


def _sport_by_game_id(session, bets: list[PaperBet]) -> dict[str, str]:
    game_ids = {bet.game_id for bet in bets}
    if not game_ids:
        return {}
    games = session.query(Game).filter(Game.game_id.in_(game_ids)).all()
    return {game.game_id: game.sport for game in games}


def _cached_live_status(session, *, sports: set[str] | None = None) -> dict:
    cache = st.session_state.get("live_status_cache")
    now_ts = time.time()
    if cache and now_ts - cache.get("fetched_at", 0) < LIVE_STATUS_TTL_SECONDS:
        return cache["data"]
    data = load_live_status_by_game_id(session, sports=sports)
    st.session_state["live_status_cache"] = {"fetched_at": now_ts, "data": data}
    return data


def _inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def _show_action_result(result, *, toast_only: bool = False) -> None:
    """Surface pipeline results as toasts so the page layout stays scrollable."""
    if result.success:
        st.toast(f"{result.title}: {result.message}", icon="✅")
    elif result.details.get("duplicate"):
        st.toast(f"{result.title}: {result.message}", icon="⚠️")
    else:
        st.toast(f"{result.title}: {result.message}", icon="❌")

    log = st.session_state.setdefault("action_log", [])
    level = "success" if result.success else ("warning" if result.details.get("duplicate") else "error")
    log.insert(0, f"{result.title}: {result.message}")
    st.session_state["action_log"] = log[:10]

    for err in result.details.get("errors") or []:
        st.toast(err, icon="⚠️")
        log.insert(0, err)
        st.session_state["action_log"] = log[:10]

    if toast_only:
        return

    # Inline alerts only when explicitly needed (e.g. persistent page-level warnings)
    if not result.success and not result.details.get("duplicate"):
        st.error(f"**{result.title}:** {result.message}")


def _render_action_log() -> None:
    log = st.session_state.get("action_log") or []
    if not log:
        return
    with st.sidebar.expander(f"Recent activity ({len(log)})", expanded=False):
        for line in log:
            st.caption(line)
        if st.button("Clear activity", use_container_width=True, key="clear_action_log"):
            st.session_state["action_log"] = []
            st.rerun()


def _bet_rows(
    session,
    bets: list[PaperBet],
    *,
    money_views: dict[int, BetMoneyView],
    live_by_game_id: dict,
    now: datetime,
    include_profit_loss: bool = False,
) -> list[dict]:
    game_cache: dict[str, Game | None] = {}
    rows = []
    for bet in bets:
        if bet.game_id not in game_cache:
            game_cache[bet.game_id] = session.query(Game).filter_by(game_id=bet.game_id).one_or_none()
        game = game_cache[bet.game_id]
        side_label = bet.side
        if game:
            side_label = game.home_team if bet.side == "home" else game.away_team
        money = money_views.get(bet.id)
        result_label = bet.outcome or "pending"
        if result_label == "pending" and game and game.home_score is not None and game.away_score is not None:
            result_label = "pending · score in, settling…"
        row = {
            "matchup": f"{game.away_team} @ {game.home_team}" if game else bet.game_id,
            "game_clock": game_clock_for(game, now=now, live_by_game_id=live_by_game_id),
            "pick": side_label,
            "odds": bet.odds_taken,
            "stake_pct": f"{bet.suggested_stake_pct:.1f}%",
            "risk": format_dollars(money.stake_dollars) if money else "—",
            "edge": f"{bet.edge_pct:+.1f}%",
            "result": result_label,
            "placed": bet.flagged_at,
        }
        if include_profit_loss:
            if money and money.profit_loss_dollars is not None:
                pl = money.profit_loss_dollars
                row["p_l"] = format_dollars(pl) if pl >= 0 else f"-{format_dollars(abs(pl))}"
            else:
                row["p_l"] = "—"
        else:
            row["to_win"] = format_dollars(money.to_win_dollars) if money else "—"
            row["payout"] = format_dollars(money.payout_dollars) if money else "—"
        rows.append(row)
    return rows


def _bet_column_config(*, include_profit_loss: bool = False) -> dict:
    config = {
        "matchup": st.column_config.TextColumn("Game", width="large"),
        "game_clock": st.column_config.TextColumn("Game clock", help=GAME_CLOCK_HELP),
        "pick": "Pick",
        "odds": "Odds",
        "stake_pct": "Stake %",
        "risk": st.column_config.TextColumn("Stake ($)", help=STAKE_DOLLARS_HELP),
        "edge": st.column_config.TextColumn("Edge", help=EDGE_COLUMN_HELP),
        "result": "Result",
        "placed": st.column_config.DatetimeColumn("Placed", format="MMM D, h:mm a"),
    }
    if include_profit_loss:
        config["p_l"] = st.column_config.TextColumn("P/L ($)", help=PROFIT_LOSS_HELP)
    else:
        config["to_win"] = st.column_config.TextColumn("To win ($)", help=TO_WIN_DOLLARS_HELP)
        config["payout"] = st.column_config.TextColumn("Payout ($)", help=PAYOUT_DOLLARS_HELP)
    return config


def _model_pick_row(pick, *, bankroll: float, live_by_game_id: dict, now: datetime) -> dict:
    preview = pick_money_preview(
        bankroll=bankroll,
        stake_pct=pick.suggested_stake_pct,
        american_odds=pick.american_odds,
    )
    live = live_by_game_id.get(pick.game_id)
    row = {
        "game": pick.matchup,
        "game_clock": format_game_clock(
            kickoff=pick.kickoff_time,
            now=now,
            live=live,
        ),
        "pick": pick.side_label,
        "odds": pick.american_odds,
        "edge": f"{pick.edge_pct:+.1f}%",
        "stake_pct": f"{pick.suggested_stake_pct:.1f}%",
        "risk": format_dollars(preview.stake_dollars),
        "to_win": format_dollars(preview.to_win_dollars),
        "type": "+EV" if pick.meets_plus_ev_threshold else "lean",
    }
    if pick.market_type == "spread" and pick.line_taken is not None:
        row["line"] = f"{pick.line_taken:+.1f}"
    return row


def _render_model_picks_table(
    session,
    *,
    sport: str,
    soft_book: str,
    days: int,
    settings,
) -> None:
    config = get_sport_config(sport)
    now = datetime.now(timezone.utc)
    picks, errors = get_model_picks(
        session, sport=sport, soft_book=soft_book, days_ahead=days, settings=settings
    )
    for err in errors[:3]:
        st.warning(err)

    bettable = [p for p in picks if p.meets_plus_ev_threshold]
    st.markdown(f"#### Model picks · {config.display_name}")
    st.caption(
        f"{config.default_market.title()} market · {len(bettable)} +EV of {len(picks)} games. "
        "Use **Log +EV only** for paper bets (recommended)."
    )

    show_leans = st.checkbox("Show leans (research only)", value=False, key=f"show_leans_{sport}")
    visible = picks if show_leans else bettable
    live_by_game_id = _cached_live_status(session, sports={sport})

    if visible:
        column_config = {
            "game_clock": st.column_config.TextColumn("Game clock", help=GAME_CLOCK_HELP),
            "edge": st.column_config.TextColumn("Edge", help=EDGE_COLUMN_HELP),
            "risk": st.column_config.TextColumn("Stake ($)", help=STAKE_DOLLARS_HELP),
            "to_win": st.column_config.TextColumn("To win ($)", help=TO_WIN_DOLLARS_HELP),
            "type": st.column_config.TextColumn("Type", help=TYPE_COLUMN_HELP),
        }
        if config.default_market == "spread":
            column_config["line"] = st.column_config.TextColumn(
                "Line",
                help="Spread taken for this side (home line shown from home perspective).",
            )
        st.dataframe(
            [
                _model_pick_row(
                    p,
                    bankroll=settings.paper_bankroll,
                    live_by_game_id=live_by_game_id,
                    now=now,
                )
                for p in visible
            ],
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Log +EV only",
                type="primary",
                use_container_width=True,
                key=f"log_ev_picks_{sport}",
                help=log_plus_ev_only_help(settings),
            ):
                _show_action_result(
                    log_model_picks(
                        sport=sport, soft_book=soft_book, days=days,
                        only_plus_ev=True, settings=settings,
                    ),
                    toast_only=True,
                )
        with c2:
            if st.button(
                "Log all picks",
                use_container_width=True,
                key=f"log_all_picks_{sport}",
                help=log_all_picks_help(settings),
            ):
                _show_action_result(
                    log_model_picks(
                        sport=sport, soft_book=soft_book, days=days,
                        only_plus_ev=False, settings=settings,
                    ),
                    toast_only=True,
                )
    elif picks:
        st.info("No +EV picks right now. Check **Show leans** or refresh odds.")
    else:
        st.info("No picks yet. Hit **Refresh odds** in the sidebar first.")


def _render_sidebar(
    session,
    settings,
    *,
    bets: list[PaperBet],
    pending_count: int,
    summary,
) -> tuple[str, int, str, str]:
    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.caption("Paper betting · research only")

    sports = list_sports()
    if "dashboard_sport" not in st.session_state:
        st.session_state.dashboard_sport = get_default_sport(
            settings=settings,
            pending_bets=bets,
            session=session,
        )
    sport = st.sidebar.selectbox(
        "Sport",
        sports,
        format_func=lambda s: get_sport_config(s).display_name,
        key="dashboard_sport",
    )
    config = get_sport_config(sport)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Your bets**")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Pending", pending_count, help=PENDING_METRIC_HELP)
    c2.metric("ROI", f"{summary.roi_pct:.1f}%", help=ROI_METRIC_HELP)

    with st.sidebar.expander("Settings", expanded=False):
        days = st.slider("Days ahead", 1, 14, 3, help=DAYS_AHEAD_HELP)
        default_strategy_index = 1
        strategy = st.selectbox(
            "Auto-flag strategy",
            ["market", "model"],
            index=default_strategy_index,
            format_func=lambda s: "ML model" if s == "model" else "Sharp vs soft",
            help=STRATEGY_HELP,
        )
        soft_book = st.selectbox(
            "Soft book",
            ["draftkings", "fanduel"],
            index=0,
            help=SOFT_BOOK_HELP,
        )

    st.sidebar.markdown("**Quick actions**")

    if st.sidebar.button(
        "Refresh odds",
        use_container_width=True,
        type="primary",
        help=REFRESH_ODDS_HELP,
    ):
        with st.spinner("Syncing fixtures and polling lines..."):
            steps = (
                sync_fixtures(sport=sport, settings=settings),
                poll_pinnacle(sport=sport, settings=settings),
                poll_softbooks(sport=sport, settings=settings),
            )
            results = list(steps)
            for step in results:
                _show_action_result(step, toast_only=True)
            if all(r.success for r in results):
                st.toast("Odds refreshed for all sources.", icon="✅")

    if st.sidebar.button(
        "Log model picks",
        use_container_width=True,
        help=sidebar_log_model_picks_help(settings),
    ):
        with st.spinner("Logging model picks..."):
            _show_action_result(
                log_model_picks(
                    sport=sport,
                    soft_book=soft_book,
                    days=days,
                    only_plus_ev=True,
                    settings=settings,
                ),
                toast_only=True,
            )

    if st.sidebar.button(
        "Settle finished games",
        use_container_width=True,
        help=SETTLE_FINISHED_GAMES_HELP,
    ):
        with st.spinner("Settling..."):
            _show_action_result(sync_scores(sport=sport, settings=settings), toast_only=True)
            _show_action_result(settle_paper_bets(settings=settings), toast_only=True)

    st.sidebar.markdown("---")
    with st.sidebar.expander("Advanced pipeline", expanded=False):
        if st.button(f"Sync {config.display_name} fixtures", use_container_width=True):
            _show_action_result(sync_fixtures(sport=sport, settings=settings))
        if st.button("Poll Pinnacle", use_container_width=True):
            _show_action_result(poll_pinnacle(sport=sport, settings=settings))
        if st.button("Poll soft books", use_container_width=True):
            _show_action_result(poll_softbooks(sport=sport, settings=settings))
        if st.button("Flag +EV only", use_container_width=True):
            _show_action_result(
                flag_opportunities(
                    sport=sport,
                    days=days,
                    strategy=strategy,
                    soft_book=soft_book,
                    settings=settings,
                )
            )
        if st.button("Full refresh + flag", use_container_width=True):
            for step_result in run_full_refresh(
                sport=sport,
                days=days,
                strategy=strategy,
                soft_book=soft_book,
                settings=settings,
            ):
                _show_action_result(step_result, toast_only=True)

    st.sidebar.markdown("---")
    _render_action_log()
    live_coverage = get_odds_coverage(sport=sport, soft_book=soft_book, settings=settings)
    missing = live_coverage.upcoming_games - live_coverage.ev_ready_games
    if missing > 0 and live_coverage.upcoming_games > 0:
        st.sidebar.caption(
            f"Odds: {live_coverage.ev_ready_games}/{live_coverage.upcoming_games} upcoming "
            f"{config.display_name} games ready on {soft_book.title()} "
            f"({missing} not loaded yet — use Refresh odds)."
        )
    else:
        st.sidebar.caption(
            f"Odds: {live_coverage.ev_ready_games}/{live_coverage.upcoming_games} upcoming "
            f"{config.display_name} games ready on {soft_book.title()}."
        )

    return sport, days, strategy, soft_book


def _render_my_bets(session, bets: list[PaperBet], summary, *, settings) -> None:
    pending = [b for b in bets if (b.outcome or "pending") == "pending"]
    settled = [b for b in bets if (b.outcome or "pending") not in (None, "pending")]

    bankroll, money_views = compute_bankroll_state(
        bets, starting_bankroll=settings.paper_bankroll
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Paper bankroll",
        format_dollars(bankroll.current_bankroll),
        help=PAPER_BANKROLL_HELP,
    )
    c2.metric("At risk", format_dollars(bankroll.at_risk), help="Total dollars on pending bets.")
    c3.metric("Available", format_dollars(bankroll.available), help="Bankroll not tied up in open bets.")
    c4.metric(
        "Settled P/L",
        format_signed_dollars(bankroll.settled_profit),
        help="Net profit or loss from settled bets so far.",
    )

    st.markdown("### My active bets")
    now = datetime.now(timezone.utc)
    sports = set(_sport_by_game_id(session, bets).values()) or {get_default_sport(settings=settings, pending_bets=bets, session=session)}
    live_by_game_id = _cached_live_status(session, sports=sports)

    if settled:
        st.caption(f"{len(settled)} settled bet(s) — expand below for results and P/L.")

    if pending:
        st.dataframe(
            _bet_rows(
                session,
                pending,
                money_views=money_views,
                live_by_game_id=live_by_game_id,
                now=now,
            ),
            use_container_width=True,
            hide_index=True,
            column_config=_bet_column_config(),
        )
    else:
        st.info("No active bets. Go to **Find picks** to log model picks or place a manual bet.")

    if settled:
        with st.expander(f"Settled bets ({len(settled)})", expanded=len(pending) == 0):
            st.dataframe(
                _bet_rows(
                    session,
                    settled[:30],
                    money_views=money_views,
                    live_by_game_id=live_by_game_id,
                    now=now,
                    include_profit_loss=True,
                ),
                use_container_width=True,
                hide_index=True,
                column_config=_bet_column_config(include_profit_loss=True),
            )


def _render_all_sports_picks_24h(session, *, soft_book: str, settings) -> None:
    """Combined +EV picks across all sports in the next 24 hours."""
    now = datetime.now(timezone.utc)
    picks, errors = get_all_sports_picks_24h(
        session, soft_book=soft_book, settings=settings
    )
    for err in errors[:5]:
        st.warning(err)

    bettable = [p for p in picks if p.meets_plus_ev_threshold]
    st.markdown("#### All sports · next 24 hours")
    st.caption(
        f"{len(bettable)} +EV pick(s) across {len({p.sport for p in picks})} sport(s) "
        f"with kickoff before {(now + timedelta(hours=24)).strftime('%a %I:%M %p %Z')}."
    )

    if not picks:
        st.info("No upcoming model picks in the next 24 hours. Use **Refresh odds** first.")
        return

    sports_for_live = {p.sport for p in picks}
    live_by_game_id = _cached_live_status(session, sports=sports_for_live)
    rows = []
    for pick in picks:
        preview = pick_money_preview(
            bankroll=settings.paper_bankroll,
            stake_pct=pick.suggested_stake_pct,
            american_odds=pick.american_odds,
        )
        live = live_by_game_id.get(pick.game_id)
        row = {
            "sport": _sport_label(pick.sport),
            "game": pick.matchup,
            "game_clock": format_game_clock(
                kickoff=pick.kickoff_time,
                now=now,
                live=live,
            ),
            "pick": pick.side_label,
            "odds": pick.american_odds,
            "edge": f"{pick.edge_pct:+.1f}%",
            "stake_pct": f"{pick.suggested_stake_pct:.1f}%",
            "risk": format_dollars(preview.stake_dollars),
            "type": "+EV" if pick.meets_plus_ev_threshold else "lean",
        }
        if pick.market_type == "spread" and pick.line_taken is not None:
            row["line"] = f"{pick.line_taken:+.1f}"
        rows.append(row)

    column_config = {
        "sport": "Sport",
        "game_clock": st.column_config.TextColumn("Game clock", help=GAME_CLOCK_HELP),
        "edge": st.column_config.TextColumn("Edge", help=EDGE_COLUMN_HELP),
        "risk": st.column_config.TextColumn("Stake ($)", help=STAKE_DOLLARS_HELP),
        "type": st.column_config.TextColumn("Type", help=TYPE_COLUMN_HELP),
    }
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def _render_find_picks(session, *, sport: str, soft_book: str, days: int, settings) -> None:
    config = get_sport_config(sport)
    now = datetime.now(timezone.utc)

    if sport_has_model(sport, settings):
        _render_model_picks_table(
            session, sport=sport, soft_book=soft_book, days=days, settings=settings
        )
    else:
        st.info(f"No ML model loaded for {config.display_name} yet.")

    st.markdown("#### Place a manual bet")
    games = (
        session.query(Game)
        .filter(Game.sport == sport, Game.kickoff_time > now)
        .order_by(Game.kickoff_time.asc())
        .limit(30)
        .all()
    )
    if not games:
        st.caption(f"No upcoming {config.display_name} games loaded.")
        return

    labels = {
        g.game_id: f"{g.away_team} @ {g.home_team} ({g.kickoff_time.astimezone().strftime('%a %I:%M %p')})"
        for g in games
    }
    with st.form("manual_bet"):
        gid = st.selectbox("Game", list(labels.keys()), format_func=lambda x: labels[x])
        game = next(g for g in games if g.game_id == gid)
        side = st.radio(
            "Side",
            ["home", "away"],
            format_func=lambda s: game.home_team if s == "home" else game.away_team,
            horizontal=True,
        )
        quote = latest_odds_quote(session, game_id=gid, book=soft_book, market_type=config.default_market)
        default_odds = -110
        if quote:
            picked = quote.odds_home if side == "home" else quote.odds_away
            if picked is not None:
                default_odds = picked
        odds = st.number_input("Odds", value=int(default_odds), step=1)
        stake = st.slider("Stake %", 0.25, 5.0, 1.0, 0.25)
        if st.form_submit_button("Place bet", type="primary", use_container_width=True):
            line = None
            if config.default_market == "spread" and quote and quote.line is not None:
                line = float(quote.line) if side == "home" else float(-quote.line)
            _show_action_result(
                place_manual_bet(
                    game_id=gid, book=soft_book, market_type=config.default_market,
                    side=side, odds_taken=int(odds), line_taken=line,
                    stake_pct=float(stake), settings=settings,
                ),
                toast_only=True,
            )


def _render_slate(session, *, sport: str, soft_book: str, settings) -> None:
    from datetime import timedelta

    config = get_sport_config(sport)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=8)
    window_end = now + timedelta(days=3)
    games = (
        session.query(Game)
        .filter(
            Game.sport == sport,
            Game.kickoff_time >= window_start,
            Game.kickoff_time <= window_end,
        )
        .order_by(Game.kickoff_time.asc())
        .limit(30)
        .all()
    )
    if not games:
        st.info("No games in the schedule window. Use **Refresh odds** in the sidebar.")
        return

    live_by_game_id = _cached_live_status(session, sports={sport})
    rows = []
    for g in games:
        pin = latest_odds_quote(session, game_id=g.game_id, book=settings.pinnacle_bookmaker, market_type=config.default_market)
        soft = latest_odds_quote(session, game_id=g.game_id, book=soft_book, market_type=config.default_market)
        rows.append(
            {
                "game_clock": game_clock_for(g, now=now, live_by_game_id=live_by_game_id),
                "kickoff": g.kickoff_time,
                "game": f"{g.away_team} @ {g.home_team}",
                "pinnacle": f"{pin.odds_home}/{pin.odds_away}" if pin and pin.odds_home else "—",
                soft_book: f"{soft.odds_home}/{soft.odds_away}" if soft and soft.odds_home else "—",
                "ready": "yes" if pin and soft else "no",
            }
        )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "game_clock": st.column_config.TextColumn("Game clock", help=GAME_CLOCK_HELP),
            "kickoff": st.column_config.DatetimeColumn("Kickoff", format="MMM D, h:mm a"),
        },
    )


def _render_performance(session, bets, summary, *, settings) -> None:
    trend = clv_trend(bets)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bets", summary.total_bets)
    c2.metric("Win rate", f"{summary.win_rate:.1f}%")
    c3.metric("ROI", f"{summary.roi_pct:.2f}%")
    avg_clv = f"{summary.avg_clv:.2f}" if summary.avg_clv is not None else "n/a"
    c4.metric("Avg CLV", avg_clv)

    sport_map = _sport_by_game_id(session, bets)
    _, money_views = compute_bankroll_state(bets, starting_bankroll=settings.paper_bankroll)
    sport_rows = compute_sport_breakdown(
        bets,
        sport_by_game_id=sport_map,
        money_views=money_views,
    )

    st.markdown("#### By sport")
    st.caption(SPORT_PERFORMANCE_CAPTION)
    if sport_rows:
        st.dataframe(
            [
                {
                    "sport": _sport_label(row.sport),
                    "bets": row.total_bets,
                    "pending": row.pending,
                    "settled": row.settled,
                    "record": f"{row.wins}-{row.losses}",
                    "win_rate": f"{row.win_rate:.1f}%",
                    "roi": f"{row.roi_pct:+.2f}%",
                    "settled_pl": format_signed_dollars(row.settled_pl_dollars),
                    "avg_clv": f"{row.avg_clv:.2f}" if row.avg_clv is not None else "n/a",
                }
                for row in sport_rows
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "sport": "Sport",
                "bets": "Bets",
                "pending": "Pending",
                "settled": "Settled",
                "record": "W-L",
                "win_rate": "Win %",
                "roi": "ROI",
                "settled_pl": st.column_config.TextColumn(
                    "Settled P/L",
                    help="Net fake-money profit or loss for settled bets in this sport.",
                ),
                "avg_clv": "Avg CLV",
            },
        )
    else:
        st.info("No paper bets yet. Log picks to start building a track record.")

    edge_rows = compute_paper_edge_buckets(
        bets,
        money_views=money_views,
        min_edge_pct=settings.min_edge_pct,
    )
    settled_edge_bets = sum(row.bets for row in edge_rows)

    st.markdown("#### By edge bucket")
    st.caption(
        f"Settled bets grouped by edge at flag time. "
        f"Rows at or above {settings.min_edge_pct:.1f}% edge are +EV buckets (highlighted)."
    )
    if settled_edge_bets:
        import pandas as pd

        table_rows = []
        for row in edge_rows:
            table_rows.append(
                {
                    "bucket": row.bucket,
                    "bets": row.bets,
                    "record": f"{row.wins}-{row.losses}" if row.bets else "—",
                    "win_rate": f"{row.win_rate:.1f}%" if row.bets else "—",
                    "roi": f"{row.roi_pct:+.2f}%" if row.bets else "—",
                    "settled_pl": format_signed_dollars(row.settled_pl_dollars) if row.bets else "—",
                    "avg_clv": f"{row.avg_clv:.2f}" if row.avg_clv is not None else "n/a",
                    "plus_ev": row.plus_ev_only,
                }
            )
        df = pd.DataFrame(table_rows)

        def _highlight_plus_ev_rows(frame: pd.DataFrame):
            styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
            for idx, plus_ev in enumerate(frame["plus_ev"]):
                if plus_ev:
                    styles.iloc[idx] = "background-color: rgba(34, 197, 94, 0.12)"
            return styles

        display = df.drop(columns=["plus_ev"])
        styled = display.style.apply(_highlight_plus_ev_rows, axis=None)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "bucket": "Edge bucket",
                "bets": "Bets",
                "record": "W-L",
                "win_rate": "Win %",
                "roi": "ROI",
                "settled_pl": st.column_config.TextColumn(
                    "Settled P/L",
                    help="Net fake-money profit or loss for settled bets in this edge bucket.",
                ),
                "avg_clv": "Avg CLV",
            },
        )
    else:
        st.caption("Edge bucket breakdown appears after bets settle.")

    st.markdown("#### CLV trend")
    if trend:
        st.line_chart(
            {
                "settled_at": [p.settled_at for p in trend],
                "bet_clv": [p.clv for p in trend],
                "cumulative_avg_clv": [p.cumulative_avg_clv for p in trend],
            },
            x="settled_at",
            y=["bet_clv", "cumulative_avg_clv"],
        )
    else:
        st.caption("CLV chart appears after you settle finished games.")


def _auto_settle_finished(settings) -> None:
    """Pull ESPN finals and mark paper bets won/lost (runs at most every 5 minutes)."""
    if time.time() - st.session_state.get("last_auto_settle_ts", 0) < 300:
        return
    st.session_state["last_auto_settle_ts"] = time.time()
    sync_scores(settings=settings)
    result = settle_paper_bets(settings=settings)
    settled = int(result.details.get("settled", 0))
    if settled > 0:
        st.toast(f"Updated {settled} bet(s) with final results.", icon="✅")
        st.rerun()


def main() -> None:
    _icon = Path(__file__).resolve().parent / "assets" / "favicon.png"
    st.set_page_config(
        page_title=f"{APP_TITLE} · Paper Bets",
        layout="wide",
        page_icon=str(_icon) if _icon.exists() else "📊",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    _render_browser_heartbeat()
    settings = get_settings()

    try:
        with session_scope(settings=settings) as session:
            bets = load_all_paper_bets(session)
            summary = compute_summary(bets)
            pending_count = summary.pending

            sport, days, strategy, soft_book = _render_sidebar(
                session,
                settings,
                bets=bets,
                pending_count=pending_count,
                summary=summary,
            )

            status = get_system_status(sport=sport, settings=settings)

            if status.degraded_sources:
                st.error(f"Degraded sources: {', '.join(status.degraded_sources)}")
            if not status.database_ok:
                st.error(status.message or "Database not connected.")
                st.info("Start Postgres with `docker compose up -d`, then refresh this page.")
                return

            st.markdown(hero_html(sport_label=_sport_label(sport)), unsafe_allow_html=True)

            tab_bets, tab_picks, tab_all, tab_slate, tab_perf = st.tabs(
                ["My bets", "Find picks", "All sports · 24h", "Schedule", "Performance"]
            )

            with tab_bets:
                _render_my_bets(session, bets, summary, settings=settings)

            with tab_picks:
                _render_find_picks(session, sport=sport, soft_book=soft_book, days=days, settings=settings)

            with tab_all:
                _render_all_sports_picks_24h(session, soft_book=soft_book, settings=settings)

            with tab_slate:
                _render_slate(session, sport=sport, soft_book=soft_book, settings=settings)

            with tab_perf:
                _render_performance(session, bets, summary, settings=settings)
    except SQLAlchemyError as exc:
        st.error(f"Database connection failed: {exc}")
        st.info("Start Postgres with `docker compose up -d` from the sports-ev-system folder, then refresh.")
        return

    _auto_settle_finished(settings)


if __name__ == "__main__":
    main()
