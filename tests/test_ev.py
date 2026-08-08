import pytest

from sports_ev.pricing import expected_value, is_plus_ev


class TestExpectedValue:
    def test_plus_money_underdog_worked_example(self):
        """
        Worked example: true prob 40%, +200, $100 stake
        Profit if win = $200
        EV = 0.40 × 200 − 0.60 × 100 = 80 − 60 = $20
        EV% = 20%
        """
        result = expected_value(true_prob=0.40, american_odds=200, stake=100)
        assert result.profit_if_win == pytest.approx(200.0, rel=1e-9)
        assert result.ev_dollars == pytest.approx(20.0, rel=1e-9)
        assert result.ev_pct == pytest.approx(20.0, rel=1e-9)
        assert result.offered_implied_prob == pytest.approx(100 / 300, rel=1e-9)
        assert result.edge_pct == pytest.approx((0.40 - 100 / 300) * 100, rel=1e-6)

    def test_minus_110_favorite_worked_example(self):
        """
        Worked example: true prob 55%, -110, $100 stake
        Profit if win = 100/1.1 = $90.909090...
        EV = 0.55 × 90.909090... − 0.45 × 100 = $5.00
        EV% = 5%
        """
        result = expected_value(true_prob=0.55, american_odds=-110, stake=100)
        assert result.profit_if_win == pytest.approx(100 / 1.1, rel=1e-9)
        assert result.ev_dollars == pytest.approx(5.0, rel=1e-6)
        assert result.ev_pct == pytest.approx(5.0, rel=1e-6)

    def test_break_even_at_implied_prob(self):
        """When true prob equals book implied prob, EV should be ~0."""
        implied = 110 / 210  # -110
        result = expected_value(true_prob=implied, american_odds=-110, stake=100)
        assert result.ev_dollars == pytest.approx(0.0, abs=1e-6)

    def test_negative_ev_when_overpriced(self):
        result = expected_value(true_prob=0.45, american_odds=-110, stake=100)
        assert result.ev_dollars < 0
        assert result.ev_pct < 0


class TestIsPlusEv:
    def test_flags_when_edge_exceeds_threshold(self):
        assert is_plus_ev(0.55, -110, min_edge_pct=2.0, stake=100) is True

    def test_rejects_when_edge_below_threshold(self):
        # Barely above implied but under 2% edge
        implied = 110 / 210
        assert is_plus_ev(implied + 0.005, -110, min_edge_pct=2.0, stake=100) is False

    def test_rejects_negative_ev(self):
        assert is_plus_ev(0.45, -110, min_edge_pct=2.0, stake=100) is False
