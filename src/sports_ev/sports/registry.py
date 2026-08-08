from __future__ import annotations

from dataclasses import dataclass

from sports_ev.config import Settings, get_settings


@dataclass(frozen=True)
class SportConfig:
    sport: str
    display_name: str
    pinnacle_league_id: int
    draftkings_event_group_id: int
    fanduel_page_id: str
    default_market: str
    espn_scoreboard_path: str
    game_id_prefix: str

    def game_id_from_matchup(self, matchup_id: int) -> str:
        return f"{self.game_id_prefix}_{matchup_id}"


def _build_registry(settings: Settings) -> dict[str, SportConfig]:
    return {
        "nfl": SportConfig(
            sport="nfl",
            display_name="NFL",
            pinnacle_league_id=settings.pinnacle_nfl_league_id,
            draftkings_event_group_id=settings.draftkings_nfl_event_group_id,
            fanduel_page_id=settings.fanduel_nfl_page_id,
            default_market="spread",
            espn_scoreboard_path="football/nfl/scoreboard",
            game_id_prefix="nfl_pin",
        ),
        "mlb": SportConfig(
            sport="mlb",
            display_name="MLB",
            pinnacle_league_id=settings.pinnacle_mlb_league_id,
            draftkings_event_group_id=settings.draftkings_mlb_event_group_id,
            fanduel_page_id=settings.fanduel_mlb_page_id,
            default_market="moneyline",
            espn_scoreboard_path="baseball/mlb/scoreboard",
            game_id_prefix="mlb_pin",
        ),
    }


def get_sport_config(sport: str, settings: Settings | None = None) -> SportConfig:
    key = sport.lower().strip()
    registry = _build_registry(settings or get_settings())
    if key not in registry:
        raise ValueError(f"Unsupported sport: {sport}")
    return registry[key]


def list_sports() -> list[str]:
    return ["nfl", "mlb"]
