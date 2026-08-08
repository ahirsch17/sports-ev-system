from __future__ import annotations

FEATURE_VERSIONS: dict[str, str] = {
    "nfl": "nfl_spread_v1",
    "mlb": "mlb_moneyline_v2",
}


def feature_version(sport: str = "nfl") -> str:
    return FEATURE_VERSIONS.get(sport.lower(), f"{sport.lower()}_v1")
