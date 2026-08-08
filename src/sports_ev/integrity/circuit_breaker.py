from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from sports_ev.config import Settings, get_settings
from sports_ev.db.models import DataQualityEvent, DataSource, OddsSnapshot, SourceStatus
from sports_ev.integrity.validators import (
    OddsSnapshotPayload,
    ValidationFailure,
    ValidationFailureReason,
    run_validation_gate,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    accepted: bool
    suppressed: bool
    source_status: str
    failures: list[ValidationFailure]
    snapshot_id: int | None = None


class CircuitBreaker:
    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    def get_or_create_source(self, source_key: str, source_type: str) -> DataSource:
        source = (
            self.session.query(DataSource)
            .filter(DataSource.source_key == source_key)
            .one_or_none()
        )
        if source is None:
            source = DataSource(
                source_key=source_key,
                source_type=source_type,
                status=SourceStatus.HEALTHY,
                consecutive_failures=0,
            )
            self.session.add(source)
            self.session.flush()
        return source

    def is_source_trusted(self, source_key: str) -> bool:
        source = (
            self.session.query(DataSource)
            .filter(DataSource.source_key == source_key)
            .one_or_none()
        )
        if source is None:
            return True
        return source.status == SourceStatus.HEALTHY

    def _log_event(
        self,
        source_key: str,
        event_type: str,
        severity: str,
        message: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        details: dict | None = None,
        parser_version: str | None = None,
    ) -> None:
        event = DataQualityEvent(
            source_key=source_key,
            event_type=event_type,
            severity=severity,
            message=message,
            expected_value=expected,
            actual_value=actual,
            details=details,
            parser_version=parser_version,
        )
        self.session.add(event)

    def _alert(self, message: str) -> None:
        logger.warning("[CIRCUIT BREAKER] %s", message)
        print(f"[CIRCUIT BREAKER ALERT] {message}")

    def record_success(
        self, source_key: str, source_type: str, parser_version: str | None = None
    ) -> DataSource:
        source = self.get_or_create_source(source_key, source_type)
        source.consecutive_failures = 0
        source.last_success_at = datetime.now(timezone.utc)
        if parser_version:
            source.parser_version = parser_version
        # Degraded sources require explicit human clear — do not auto-recover here.
        return source

    def record_failure(
        self,
        source_key: str,
        source_type: str,
        failures: list[ValidationFailure],
        *,
        parser_version: str | None = None,
    ) -> DataSource:
        source = self.get_or_create_source(source_key, source_type)
        source.consecutive_failures += 1
        source.last_failure_at = datetime.now(timezone.utc)

        for failure in failures:
            self._log_event(
                source_key,
                f"validation_{failure.reason}",
                "error",
                failure.message,
                expected=failure.expected,
                actual=failure.actual,
                parser_version=parser_version,
            )

        threshold = self.settings.circuit_breaker_failure_threshold
        if source.consecutive_failures >= threshold and source.status != SourceStatus.DEGRADED:
            source.status = SourceStatus.DEGRADED
            source.degraded_at = datetime.now(timezone.utc)
            self._log_event(
                source_key,
                "circuit_breaker_tripped",
                "circuit_break",
                f"Source {source_key} marked degraded after {source.consecutive_failures} consecutive failures",
                details={"failure_count": source.consecutive_failures, "threshold": threshold},
            )
            self._alert(
                f"Source {source_key} DEGRADED after {source.consecutive_failures} failures — "
                "opportunities from this source will be suppressed until human review"
            )

        return source

    def clear_degraded(self, source_key: str, *, reviewed_by: str = "human") -> DataSource:
        source = self.get_or_create_source(source_key, "unknown")
        source.status = SourceStatus.HEALTHY
        source.consecutive_failures = 0
        source.degraded_at = None
        self._log_event(
            source_key,
            "manual_clear",
            "warning",
            f"Degraded status cleared by {reviewed_by}",
        )
        return source

    def _latest_line(
        self, game_id: str, book: str, market_type: str
    ) -> float | None:
        row = (
            self.session.query(OddsSnapshot)
            .filter_by(game_id=game_id, book=book, market_type=market_type)
            .order_by(OddsSnapshot.captured_at.desc())
            .first()
        )
        return row.line if row else None

    def ingest_odds_snapshot(
        self,
        payload: OddsSnapshotPayload,
        *,
        previous_line: float | None = None,
        skip_freshness: bool = False,
    ) -> IngestionResult:
        source_key = f"{payload.book}:{payload.market_type}"

        if not self.is_source_trusted(source_key):
            return IngestionResult(
                accepted=False,
                suppressed=True,
                source_status=SourceStatus.DEGRADED,
                failures=[
                    ValidationFailure(
                        reason=ValidationFailureReason.SCHEMA,
                        message="Source is degraded; snapshot rejected pending human review",
                    )
                ],
            )

        if previous_line is None:
            previous_line = self._latest_line(
                payload.game_id, payload.book, payload.market_type
            )

        failures = run_validation_gate(
            payload,
            previous_line=previous_line,
            max_stale_seconds=self.settings.circuit_breaker_max_stale_seconds,
            max_line_jump=self.settings.circuit_breaker_max_line_jump,
            skip_freshness=skip_freshness,
        )

        if failures:
            source = self.record_failure(
                source_key,
                payload.source,
                failures,
                parser_version=payload.parser_version,
            )
            return IngestionResult(
                accepted=False,
                suppressed=source.status == SourceStatus.DEGRADED,
                source_status=source.status,
                failures=failures,
            )

        source = self.record_success(source_key, payload.source, payload.parser_version)

        snapshot = OddsSnapshot(
            game_id=payload.game_id,
            book=payload.book,
            market_type=payload.market_type,
            captured_at=payload.captured_at,
            odds_home=payload.odds_home,
            odds_away=payload.odds_away,
            line=payload.line,
            source=payload.source,
            parser_version=payload.parser_version,
            raw_response_hash=payload.raw_response_hash,
        )
        self.session.add(snapshot)
        self.session.flush()

        return IngestionResult(
            accepted=True,
            suppressed=False,
            source_status=source.status,
            failures=[],
            snapshot_id=snapshot.id,
        )
