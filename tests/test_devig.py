import pytest

from sports_ev.pricing import (
    DevigMethod,
    devig_two_way,
    implied_probs_from_american,
    multiplicative_devig,
    overround,
    shin_devig,
)


class TestMultiplicativeDevig:
    def test_symmetric_minus_110(self):
        """
        Worked example: -110 / -110
        Raw implied each = 110/210 = 0.5238095238
        Overround = 1.0476190476
        Fair each = 0.5
        """
        raw_home, raw_away = implied_probs_from_american(-110, -110)
        assert raw_home == pytest.approx(110 / 210, rel=1e-9)
        assert overround([raw_home, raw_away]) == pytest.approx(220 / 210, rel=1e-9)

        fair = multiplicative_devig([raw_home, raw_away])
        assert fair[0] == pytest.approx(0.5, rel=1e-9)
        assert fair[1] == pytest.approx(0.5, rel=1e-9)
        assert sum(fair) == pytest.approx(1.0, rel=1e-9)

    def test_asymmetric_minus_150_plus_130(self):
        """
        Worked example: -150 / +130
        Home implied = 150/250 = 0.6
        Away implied = 100/230 = 0.4347826087
        Overround = 1.0347826087
        Fair home = 0.6 / 1.0347826087 = 0.5798319328
        Fair away = 0.4347826087 / 1.0347826087 = 0.4201680672
        """
        raw_home, raw_away = implied_probs_from_american(-150, 130)
        assert raw_home == pytest.approx(0.6, rel=1e-9)
        assert raw_away == pytest.approx(100 / 230, rel=1e-9)

        fair_home, fair_away = multiplicative_devig([raw_home, raw_away])
        assert fair_home == pytest.approx(0.6 / (0.6 + 100 / 230), rel=1e-6)
        assert fair_away == pytest.approx((100 / 230) / (0.6 + 100 / 230), rel=1e-6)
        assert fair_home + fair_away == pytest.approx(1.0, rel=1e-9)


class TestShinDevig:
    def test_symmetric_minus_110(self):
        fair = shin_devig(implied_probs_from_american(-110, -110))
        assert fair[0] == pytest.approx(0.5, rel=1e-4)
        assert fair[1] == pytest.approx(0.5, rel=1e-4)
        assert sum(fair) == pytest.approx(1.0, rel=1e-4)

    def test_shin_differs_from_multiplicative_on_asymmetric_line(self):
        raw = list(implied_probs_from_american(-150, 130))
        mult = multiplicative_devig(raw)
        shin = shin_devig(raw)
        # Shin allocates slightly more to the favorite on asymmetric markets
        assert shin[0] > mult[0]
        assert shin[1] < mult[1]
        assert sum(shin) == pytest.approx(1.0, rel=1e-6)


class TestDevigTwoWay:
    def test_toggle_multiplicative(self):
        home, away = devig_two_way(-110, -110, method=DevigMethod.MULTIPLICATIVE)
        assert home == pytest.approx(0.5, rel=1e-9)
        assert away == pytest.approx(0.5, rel=1e-9)

    def test_toggle_shin(self):
        home, away = devig_two_way(-110, -110, method=DevigMethod.SHIN)
        assert home == pytest.approx(0.5, rel=1e-4)
        assert away == pytest.approx(0.5, rel=1e-4)
