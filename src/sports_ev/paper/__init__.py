"""Paper bet tracking — flag opportunities, settle outcomes, aggregate metrics."""

from sports_ev.paper.flagging import FlagResult, OpportunityFlagger
from sports_ev.paper.kelly import expected_profit_pct, kelly_fraction
from sports_ev.paper.placement import ManualBetPlacer, ManualBetRequest, OddsQuote, PlacementResult, latest_odds_quote
from sports_ev.paper.metrics import ClvTrendPoint, PaperBetSummary, clv_trend, compute_summary, load_all_paper_bets
from sports_ev.paper.settlement import PaperBetSettlementService, SettlementResult

__all__ = [
    "ClvTrendPoint",
    "FlagResult",
    "OpportunityFlagger",
    "PaperBetSettlementService",
    "PaperBetSummary",
    "SettlementResult",
    "clv_trend",
    "compute_summary",
    "expected_profit_pct",
    "kelly_fraction",
    "load_all_paper_bets",
    "latest_odds_quote",
    "ManualBetPlacer",
    "ManualBetRequest",
    "OddsQuote",
    "PlacementResult",
]
