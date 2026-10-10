"""Smoke tests for odds math and edge helpers."""

from nfl_edge.odds_math import (
    american_to_implied,
    expected_value,
    implied_to_american,
    margin_to_home_win_prob,
    prob_over_line,
)


def test_american_roundtrip():
    p = american_to_implied(-110)
    assert 0.52 < p < 0.53
    assert implied_to_american(0.5) == 100 or implied_to_american(0.5) == -100


def test_ev_favorite():
    # model 60% at -110 should be +EV
    assert expected_value(0.60, -110) > 0


def test_margin_and_totals():
    assert margin_to_home_win_prob(0.0) == 0.5
    assert 0.49 < prob_over_line(45.0, 10.0, 45.0) < 0.51
