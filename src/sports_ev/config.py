from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://sports_ev:sports_ev_dev@localhost:5433/sports_ev"
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_max_line_jump: float = 15.0
    circuit_breaker_max_stale_seconds: int = 3600

    pinnacle_bookmaker: str = "pinnacle"
    pinnacle_nfl_league_id: int = 889
    pinnacle_mlb_league_id: int = 246
    pinnacle_football_sport_id: int = 15

    default_devig_method: str = "multiplicative"
    min_edge_pct: float = 2.5

    softbook_min_interval_seconds: float = 2.0
    draftkings_nfl_event_group_id: int = 88808
    draftkings_mlb_event_group_id: int = 84240
    fanduel_api_base_url: str = "https://sbapi.nj.sportsbook.fanduel.com"
    fanduel_nfl_page_id: str = "nfl"
    fanduel_mlb_page_id: str = "mlb"

    model_artifact_path: str = "artifacts/nfl_spread_lgbm_v1.joblib"
    mlb_model_artifact_path: str = "artifacts/mlb_moneyline_lgbm_v3.joblib"
    model_min_train_games: int = 20
    mlb_model_min_train_games: int = 30

    kelly_fraction: float = 0.25
    max_stake_pct: float = 2.0
    paper_bankroll: float = 1000.0


def get_settings() -> Settings:
    return Settings()
