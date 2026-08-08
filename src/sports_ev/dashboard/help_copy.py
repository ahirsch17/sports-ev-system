"""Dashboard tooltip and help text."""

from __future__ import annotations

from sports_ev.config import Settings


def plus_ev_threshold_pct(settings: Settings) -> float:
    return settings.min_edge_pct


def log_all_picks_help(settings: Settings) -> str:
    pct = plus_ev_threshold_pct(settings)
    return (
        f"Logs every model pick for upcoming games, including leans below {pct:.1f}% edge. "
        "Updates an existing pending pick for the same game if one exists."
    )


def log_plus_ev_only_help(settings: Settings) -> str:
    pct = plus_ev_threshold_pct(settings)
    return (
        f"Logs only picks with edge at or above {pct:.1f}% (+EV). "
        f"Skips leans below that threshold and duplicate pending bets."
    )


def sidebar_log_model_picks_help(settings: Settings) -> str:
    pct = plus_ev_threshold_pct(settings)
    return (
        f"Logs only +EV model picks (edge ≥ {pct:.1f}%). "
        f"Use Find picks → Log all picks if you also want leans below that bar."
    )


PENDING_METRIC_HELP = "Paper bets waiting for the game to finish and settle."
ROI_METRIC_HELP = "Return on investment across all settled paper bets so far."
PAPER_BANKROLL_HELP = (
    "Simulated fake money for research only. Dollar stakes and payouts are calculated from "
    "your paper bankroll and each bet's suggested stake percentage. Not real money."
)
STAKE_DOLLARS_HELP = "Dollar amount risked on this bet: current paper bankroll times suggested stake %."
TO_WIN_DOLLARS_HELP = "Profit if the bet wins, based on American odds (stake not included)."
PAYOUT_DOLLARS_HELP = "Total return if the bet wins: stake plus profit."
PROFIT_LOSS_HELP = "Actual profit or loss in dollars after the bet settled."

DAYS_AHEAD_HELP = "How many days of upcoming games to scan for model picks and auto-flagging."
STRATEGY_HELP = (
    "Sharp vs soft compares Pinnacle to your soft book. ML model uses the trained model for picks."
)
SOFT_BOOK_HELP = "Sportsbook used for soft lines, model picks, and paper bet placement."

REFRESH_ODDS_HELP = "Syncs fixtures from Pinnacle, then polls sharp and soft book lines."
SETTLE_FINISHED_GAMES_HELP = (
    "Pulls final scores from ESPN, then marks pending paper bets as win, loss, or push. "
    "The dashboard also does this automatically every few minutes."
)

MODEL_PICKS_CAPTION = (
    "One pick per game: the model's preferred side at current soft-book odds. "
    "Hover column headers for edge and type details."
)

EDGE_COLUMN_HELP = "Expected value edge: model win probability minus the book's implied probability."
TYPE_COLUMN_HELP = (
    "+EV means edge at or above threshold, positive Kelly stake, and valid line data. "
    "Lean is the model's best side but fails one of those checks."
)

SPORT_PERFORMANCE_CAPTION = (
    "Track model results by sport as you add NFL, MLB, and others. "
    "Settled P/L uses your paper bankroll and stake sizes at bet time."
)

SCOPE_CAPTION = (
    "Pre-game only today: NFL spreads and MLB moneylines. No live betting, props, "
    "or auto hedge/sell yet. Re-run Refresh odds or Log picks to update lines on open paper bets."
)

GAME_CLOCK_HELP = (
    "Countdown before kickoff, live score and inning/quarter from ESPN, or final score when done."
)
