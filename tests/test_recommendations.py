from __future__ import annotations

from datetime import datetime, timezone

from sports_ev.config import Settings
from sports_ev.paper.recommendations import ModelPick, pick_is_bettable


def _sample_pick(**overrides) -> ModelPick:
    base = dict(
        game_id="g1",
        sport="mlb",
        matchup="A @ B",
        kickoff_time=datetime.now(timezone.utc),
        side="home",
        side_label="B",
        american_odds=-110,
        line_taken=None,
        market_type="moneyline",
        model_prob=0.55,
        market_implied_prob=0.52,
        edge_pct=3.0,
        suggested_stake_pct=1.5,
        model_version="test",
        meets_plus_ev_threshold=True,
    )
    base.update(overrides)
    return ModelPick(**base)


class TestPickIsBettable:
    def test_positive_edge_and_stake(self):
        pick = _sample_pick()
        assert pick_is_bettable(pick, Settings(min_edge_pct=2.5)) is True

    def test_rejects_negative_edge(self):
        pick = _sample_pick(edge_pct=-1.0, meets_plus_ev_threshold=False)
        assert pick_is_bettable(pick, Settings(min_edge_pct=2.5)) is False

    def test_rejects_zero_kelly_stake(self):
        pick = _sample_pick(suggested_stake_pct=0.0, meets_plus_ev_threshold=False)
        assert pick_is_bettable(pick, Settings(min_edge_pct=2.5)) is False

    def test_spread_requires_line(self):
        pick = _sample_pick(market_type="spread", line_taken=None, meets_plus_ev_threshold=False)
        assert pick_is_bettable(pick, Settings(min_edge_pct=2.5)) is False

    def test_spread_with_line(self):
        pick = _sample_pick(market_type="spread", line_taken=-3.5)
        assert pick_is_bettable(pick, Settings(min_edge_pct=2.5)) is True


class TestSpreadSettlementLine:
    def test_away_spread_uses_home_line_from_stored_side_line(self):
        from sports_ev.backtest.models import BetOutcome
        from sports_ev.paper.settlement import _spread_outcome

        # Home -3.5, away took +3.5 (stored as line_taken=+3.5). Final 20-17 home wins by 3.
        outcome = _spread_outcome("away", home_score=20, away_score=17, line_taken=3.5)
        assert outcome == BetOutcome.WIN

        outcome_home = _spread_outcome("home", home_score=20, away_score=17, line_taken=-3.5)
        assert outcome_home == BetOutcome.LOSS
