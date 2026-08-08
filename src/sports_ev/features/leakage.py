from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class LeakageError(Exception):
    """Raised when a feature uses data not knowable before kickoff."""


@dataclass(frozen=True)
class SourcedValue:
    name: str
    value: float
    known_at: datetime
    source: str


@dataclass
class FeatureAudit:
    kickoff_time: datetime
    sources: list[SourcedValue] = field(default_factory=list)

    def add(self, name: str, value: float, known_at: datetime, source: str) -> None:
        self.sources.append(SourcedValue(name=name, value=value, known_at=known_at, source=source))

    def latest_known_at(self) -> datetime | None:
        if not self.sources:
            return None
        return max(s.known_at for s in self.sources)


def assert_no_leakage(audit: FeatureAudit) -> None:
    """Fail if any sourced value was not knowable before kickoff."""
    for sourced in audit.sources:
        if sourced.known_at >= audit.kickoff_time:
            raise LeakageError(
                f"Feature {sourced.name!r} used data from {sourced.source} "
                f"known at {sourced.known_at.isoformat()} which is not before "
                f"kickoff {audit.kickoff_time.isoformat()}"
            )
