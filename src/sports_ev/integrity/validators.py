from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable


class ValidationFailureReason(StrEnum):
    SCHEMA = "schema"
    SANITY = "sanity"
    FRESHNESS = "freshness"


@dataclass(frozen=True)
class ValidationFailure:
    reason: ValidationFailureReason
    message: str
    expected: str | None = None
    actual: str | None = None


@dataclass
class OddsSnapshotPayload:
    game_id: str
    book: str
    market_type: str
    captured_at: datetime
    odds_home: int | None = None
    odds_away: int | None = None
    line: float | None = None
    source: str = "api"
    parser_version: str | None = None
    raw_response_hash: str | None = None


# American odds bounds per spec
MIN_AMERICAN_ODDS = -10000
MAX_AMERICAN_ODDS = 10000
MIN_SPREAD = -60.0
MAX_SPREAD = 60.0


def _is_valid_american_odds(value: int | None) -> bool:
    if value is None:
        return True
    if value == 0:
        return False
    return MIN_AMERICAN_ODDS <= value <= MAX_AMERICAN_ODDS


def validate_schema(payload: OddsSnapshotPayload) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []

    required_strings = {
        "game_id": payload.game_id,
        "book": payload.book,
        "market_type": payload.market_type,
        "source": payload.source,
    }
    for field_name, value in required_strings.items():
        if not value or not str(value).strip():
            failures.append(
                ValidationFailure(
                    reason=ValidationFailureReason.SCHEMA,
                    message=f"Missing required field: {field_name}",
                    expected="non-empty string",
                    actual=repr(value),
                )
            )

    if payload.captured_at is None:
        failures.append(
            ValidationFailure(
                reason=ValidationFailureReason.SCHEMA,
                message="Missing captured_at timestamp",
                expected="datetime",
                actual="None",
            )
        )

    if payload.market_type in ("spread", "moneyline", "total"):
        if payload.odds_home is None and payload.odds_away is None:
            failures.append(
                ValidationFailure(
                    reason=ValidationFailureReason.SCHEMA,
                    message="At least one odds value required",
                    expected="odds_home or odds_away",
                    actual="both None",
                )
            )

    if payload.market_type == "spread" and payload.line is None:
        failures.append(
            ValidationFailure(
                reason=ValidationFailureReason.SCHEMA,
                message="Spread market requires line",
                expected="float",
                actual="None",
            )
        )

    return failures


def validate_sanity(payload: OddsSnapshotPayload) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []

    for label, odds in (("odds_home", payload.odds_home), ("odds_away", payload.odds_away)):
        if odds is not None and not _is_valid_american_odds(odds):
            failures.append(
                ValidationFailure(
                    reason=ValidationFailureReason.SANITY,
                    message=f"{label} out of range",
                    expected=f"{MIN_AMERICAN_ODDS} to {MAX_AMERICAN_ODDS}",
                    actual=str(odds),
                )
            )

    if payload.line is not None and not (MIN_SPREAD <= payload.line <= MAX_SPREAD):
        failures.append(
            ValidationFailure(
                reason=ValidationFailureReason.SANITY,
                message="Spread line out of range",
                expected=f"{MIN_SPREAD} to {MAX_SPREAD}",
                actual=str(payload.line),
            )
        )

    return failures


def validate_freshness(
    payload: OddsSnapshotPayload,
    *,
    now: datetime | None = None,
    max_stale_seconds: int = 3600,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    now = now or datetime.now(timezone.utc)
    captured = payload.captured_at
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)

    age_seconds = (now - captured).total_seconds()
    if age_seconds > max_stale_seconds:
        failures.append(
            ValidationFailure(
                reason=ValidationFailureReason.FRESHNESS,
                message="Data is stale",
                expected=f"<= {max_stale_seconds}s old",
                actual=f"{age_seconds:.0f}s old",
            )
        )

    if captured > now:
        failures.append(
            ValidationFailure(
                reason=ValidationFailureReason.FRESHNESS,
                message="Captured timestamp is in the future",
                expected=f"<= {now.isoformat()}",
                actual=captured.isoformat(),
            )
        )

    return failures


def validate_line_jump(
    payload: OddsSnapshotPayload,
    previous_line: float | None,
    *,
    max_jump: float = 15.0,
) -> list[ValidationFailure]:
    if payload.line is None or previous_line is None:
        return []

    jump = abs(payload.line - previous_line)
    if jump > max_jump:
        return [
            ValidationFailure(
                reason=ValidationFailureReason.SANITY,
                message="Line jump exceeds threshold",
                expected=f"jump <= {max_jump}",
                actual=f"jump={jump:.1f} ({previous_line} -> {payload.line})",
            )
        ]
    return []


def run_validation_gate(
    payload: OddsSnapshotPayload,
    *,
    previous_line: float | None = None,
    now: datetime | None = None,
    max_stale_seconds: int = 3600,
    max_line_jump: float = 15.0,
    skip_freshness: bool = False,
) -> list[ValidationFailure]:
    failures: list[ValidationFailure] = []
    failures.extend(validate_schema(payload))
    if failures:
        return failures
    failures.extend(validate_sanity(payload))
    if not skip_freshness:
        failures.extend(
            validate_freshness(payload, now=now, max_stale_seconds=max_stale_seconds)
        )
    failures.extend(validate_line_jump(payload, previous_line, max_jump=max_line_jump))
    return failures
