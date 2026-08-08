from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from sports_ev.config import Settings, get_settings
from sports_ev.db.session import create_db_engine

# Idempotent column additions for existing databases (create_all skips alters).
_GAME_CLOSING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("pinnacle_closing_spread", "DOUBLE PRECISION"),
    ("pinnacle_closing_home_odds", "INTEGER"),
    ("pinnacle_closing_away_odds", "INTEGER"),
    ("closing_captured_at", "TIMESTAMPTZ"),
)


def migrate_db(engine: Engine | None = None, settings: Settings | None = None) -> list[str]:
    """Apply additive schema updates to an existing database."""
    engine = engine or create_db_engine(settings)
    dialect = engine.dialect.name
    col_types = _column_types_for(dialect)
    applied: list[str] = []
    with engine.begin() as conn:
        existing = {col["name"] for col in inspect(conn).get_columns("games")}
        for name, _ in _GAME_CLOSING_COLUMNS:
            if name in existing:
                continue
            col_type = col_types[name]
            conn.execute(text(f"ALTER TABLE games ADD COLUMN {name} {col_type}"))
            applied.append(f"games.{name}")
    return applied


def _column_types_for(dialect: str) -> dict[str, str]:
    if dialect == "sqlite":
        return {
            "pinnacle_closing_spread": "FLOAT",
            "pinnacle_closing_home_odds": "INTEGER",
            "pinnacle_closing_away_odds": "INTEGER",
            "closing_captured_at": "DATETIME",
        }
    return {name: col_type for name, col_type in _GAME_CLOSING_COLUMNS}
