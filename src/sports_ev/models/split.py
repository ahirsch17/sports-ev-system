from __future__ import annotations

from dataclasses import dataclass

from sports_ev.models.dataset import LabeledRow


@dataclass(frozen=True)
class TimeSplit:
    train: list[LabeledRow]
    validation: list[LabeledRow]


def time_ordered_split(rows: list[LabeledRow], *, val_fraction: float = 0.2) -> TimeSplit:
    """
    Split rows chronologically — train on earlier games, validate on later ones.
    Never shuffles; preserves temporal order.
    """
    if not rows:
        return TimeSplit(train=[], validation=[])

    ordered = sorted(rows, key=lambda r: r.kickoff_time)
    if len(ordered) < 2:
        return TimeSplit(train=ordered, validation=[])

    val_count = max(1, int(len(ordered) * val_fraction))
    if val_count >= len(ordered):
        val_count = 1

    split_at = len(ordered) - val_count
    if split_at < 1:
        split_at = 1

    return TimeSplit(train=ordered[:split_at], validation=ordered[split_at:])
