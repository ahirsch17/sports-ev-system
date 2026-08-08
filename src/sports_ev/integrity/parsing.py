from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


ParseFn = Callable[[Any], Any]


@dataclass
class ParseStrategy:
    name: str
    version: str
    parse: ParseFn


@dataclass
class ParseResult:
    value: Any
    strategy_name: str
    parser_version: str


class AdaptiveParser:
    """Try multiple parsing strategies in order; log which version succeeded."""

    def __init__(self, field_name: str, strategies: list[ParseStrategy]):
        if not strategies:
            raise ValueError("At least one parse strategy required")
        self.field_name = field_name
        self.strategies = strategies

    @property
    def parser_version(self) -> str:
        return "+".join(f"{s.name}@{s.version}" for s in self.strategies)

    def parse(self, raw: Any) -> ParseResult:
        errors: list[str] = []
        for strategy in self.strategies:
            try:
                value = strategy.parse(raw)
                if value is None:
                    errors.append(f"{strategy.name}: returned None")
                    continue
                return ParseResult(
                    value=value,
                    strategy_name=strategy.name,
                    parser_version=strategy.version,
                )
            except Exception as exc:  # noqa: BLE001 — collect all strategy failures
                errors.append(f"{strategy.name}: {exc}")

        raise ValueError(
            f"All parse strategies failed for {self.field_name}: {'; '.join(errors)}"
        )


def parse_json_api_path(data: dict[str, Any], path: list[str]) -> ParseFn:
    def _parse(_raw: Any) -> Any:
        node = data if _raw is None else _raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Missing key {key!r} at path {path}")
            node = node[key]
        return node

    return _parse


def parse_css_regex(pattern: str, group: int = 1, cast: Callable[[str], Any] = str) -> ParseFn:
    compiled = re.compile(pattern)

    def _parse(raw: Any) -> Any:
        text = raw if isinstance(raw, str) else str(raw)
        match = compiled.search(text)
        if not match:
            raise ValueError(f"Pattern {pattern!r} not found")
        return cast(match.group(group))

    return _parse


def parse_american_odds(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    text = str(raw).strip().replace(",", "")
    if text.startswith("+"):
        text = text[1:]
    value = int(text)
    if value == 0:
        raise ValueError("American odds cannot be zero")
    return value


def parse_spread_line(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    return float(text)


def hash_raw_response(raw: Any) -> str:
    if isinstance(raw, (dict, list)):
        payload = json.dumps(raw, sort_keys=True, default=str)
    else:
        payload = str(raw)
    return hashlib.sha256(payload.encode()).hexdigest()
