from __future__ import annotations

from ui.board import build_game_board


def _h2h(selection: str, price: int, model_prob: float, ev: float) -> dict:
    return {
        "game_id": "g1",
        "market": "h2h",
        "selection": selection,
        "home_team": "Los Angeles Rams",
        "away_team": "Buffalo Bills",
        "dk_price": price,
        "model_prob": model_prob,
        "ev": ev,
        "passes_threshold": False,
    }


def test_winner_card_uses_picked_teams_price():
    game = {
        "game_id": "g1",
        "away_team": "BUF",
        "home_team": "LA",
        "pred_margin": -0.3,
        "pred_total": 57.5,
        "home_win_prob": 0.49,
    }
    edges = [
        _h2h("Los Angeles Rams", -180, 0.60, -0.07),
        _h2h("Buffalo Bills", 150, 0.40, -0.005),
    ]
    card = build_game_board(sport="NFL", games=[game], advice={}, market_edges=edges)[0]
    assert card["winner"]["pick"] == "BUF"
    assert card["winner"]["dk_price"] == 150
    assert "40%" in card["winner"]["stake"]
