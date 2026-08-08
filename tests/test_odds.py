import pytest

from sports_ev.pricing.odds import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    profit_if_win,
)


class TestAmericanImpliedProb:
    def test_negative_odds_favorite(self):
        # -110 → 110/(110+100) = 0.5238095238
        assert american_to_implied_prob(-110) == pytest.approx(110 / 210, rel=1e-9)

    def test_positive_odds_underdog(self):
        # +150 → 100/(150+100) = 0.4
        assert american_to_implied_prob(150) == pytest.approx(0.4, rel=1e-9)

    def test_even_money(self):
        assert american_to_implied_prob(100) == pytest.approx(0.5, rel=1e-9)


class TestOddsConversions:
    def test_american_decimal_roundtrip(self):
        for american in (-110, -150, 130, 200):
            assert decimal_to_american(american_to_decimal(american)) == american

    def test_profit_if_win_negative_odds(self):
        # $100 at -110 wins $90.909...
        assert profit_if_win(100, -110) == pytest.approx(100 / 1.1, rel=1e-9)

    def test_profit_if_win_positive_odds(self):
        assert profit_if_win(100, 200) == pytest.approx(200.0, rel=1e-9)

    def test_zero_odds_raises(self):
        with pytest.raises(ValueError):
            american_to_implied_prob(0)
