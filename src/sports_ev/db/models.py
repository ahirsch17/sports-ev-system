from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SourceStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    oddspapi_fixture_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    sport: Mapped[str] = mapped_column(String(32), nullable=False)
    home_team: Mapped[str] = mapped_column(String(64), nullable=False)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False)
    kickoff_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int | None] = mapped_column(Integer)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    pinnacle_closing_spread: Mapped[float | None] = mapped_column(Float)
    pinnacle_closing_home_odds: Mapped[int | None] = mapped_column(Integer)
    pinnacle_closing_away_odds: Mapped[int | None] = mapped_column(Integer)
    closing_captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    odds_snapshots: Mapped[list[OddsSnapshot]] = relationship(back_populates="game")
    team_game_stats: Mapped[list[TeamGameStat]] = relationship(back_populates="game")
    injury_reports: Mapped[list[InjuryReport]] = relationship(back_populates="game")
    model_predictions: Mapped[list[ModelPrediction]] = relationship(back_populates="game")
    paper_bets: Mapped[list[PaperBet]] = relationship(back_populates="game")

    __table_args__ = (
        Index("ix_games_sport_season_week", "sport", "season", "week"),
        Index("ix_games_kickoff_time", "kickoff_time"),
    )


class DataSource(Base):
    """Tracks health of each odds/data source for circuit breaker."""

    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)  # api, scraped
    status: Mapped[str] = mapped_column(String(16), default=SourceStatus.HEALTHY, nullable=False)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parser_version: Mapped[str | None] = mapped_column(String(32))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    degraded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), ForeignKey("games.game_id"), nullable=False)
    book: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    odds_home: Mapped[int | None] = mapped_column(Integer)
    odds_away: Mapped[int | None] = mapped_column(Integer)
    line: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16), nullable=False)  # scraped, api
    parser_version: Mapped[str | None] = mapped_column(String(32))
    raw_response_hash: Mapped[str | None] = mapped_column(String(64))
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="odds_snapshots")

    __table_args__ = (
        Index("ix_odds_snapshots_game_book_market_time", "game_id", "book", "market_type", "captured_at"),
    )


class TeamGameStat(Base):
    __tablename__ = "team_game_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), ForeignKey("games.game_id"), nullable=False)
    team: Mapped[str] = mapped_column(String(64), nullable=False)
    stat_key: Mapped[str] = mapped_column(String(64), nullable=False)
    stat_value: Mapped[float] = mapped_column(Float, nullable=False)
    known_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="team_game_stats")

    __table_args__ = (
        Index("ix_team_game_stats_game_team", "game_id", "team"),
        Index("ix_team_game_stats_known_at", "known_at"),
    )


class InjuryReport(Base):
    __tablename__ = "injury_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), ForeignKey("games.game_id"), nullable=False)
    team: Mapped[str] = mapped_column(String(64), nullable=False)
    player: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    report_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="injury_reports")

    __table_args__ = (Index("ix_injury_reports_game_team", "game_id", "team"),)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), ForeignKey("games.game_id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_prob: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    shap_values: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="model_predictions")

    __table_args__ = (
        Index("ix_model_predictions_game_version", "game_id", "model_version", "predicted_at"),
    )


class PaperBet(Base):
    __tablename__ = "paper_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(64), ForeignKey("games.game_id"), nullable=False)
    book: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(32), nullable=False)
    flagged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    suggested_stake_pct: Mapped[float] = mapped_column(Float, nullable=False)
    odds_taken: Mapped[int] = mapped_column(Integer, nullable=False)
    line_taken: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    edge_pct: Mapped[float] = mapped_column(Float, nullable=False)
    audit_trail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(16))  # win, loss, push, pending
    closing_line: Mapped[float | None] = mapped_column(Float)
    clv: Mapped[float | None] = mapped_column(Float)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[Game] = relationship(back_populates="paper_bets")

    __table_args__ = (Index("ix_paper_bets_flagged_at", "flagged_at"),)


class DataQualityEvent(Base):
    __tablename__ = "data_quality_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_key: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)  # warning, error, circuit_break
    message: Mapped[str] = mapped_column(Text, nullable=False)
    expected_value: Mapped[str | None] = mapped_column(Text)
    actual_value: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    parser_version: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_data_quality_events_source_created", "source_key", "created_at"),
        Index("ix_data_quality_events_event_type", "event_type"),
    )
