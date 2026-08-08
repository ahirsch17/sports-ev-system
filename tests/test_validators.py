from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sports_ev.integrity.parsing import (
    AdaptiveParser,
    ParseStrategy,
    parse_american_odds,
    parse_css_regex,
    parse_json_api_path,
    parse_spread_line,
)
from sports_ev.integrity.validators import (
    OddsSnapshotPayload,
    run_validation_gate,
)


class TestValidators:
    def test_valid_spread_passes(self):
        payload = OddsSnapshotPayload(
            game_id="g1",
            book="pinnacle",
            market_type="spread",
            captured_at=datetime.now(timezone.utc),
            odds_home=-105,
            odds_away=-115,
            line=-2.5,
            source="api",
        )
        assert run_validation_gate(payload) == []

    def test_missing_line_fails_schema(self):
        payload = OddsSnapshotPayload(
            game_id="g1",
            book="pinnacle",
            market_type="spread",
            captured_at=datetime.now(timezone.utc),
            odds_home=-105,
            source="api",
        )
        failures = run_validation_gate(payload)
        assert any("Spread market requires line" in f.message for f in failures)

    def test_extreme_odds_fail_sanity(self):
        payload = OddsSnapshotPayload(
            game_id="g1",
            book="pinnacle",
            market_type="moneyline",
            captured_at=datetime.now(timezone.utc),
            odds_home=50000,
            source="api",
        )
        failures = run_validation_gate(payload)
        assert any("out of range" in f.message for f in failures)

    def test_line_jump_detected(self):
        payload = OddsSnapshotPayload(
            game_id="g1",
            book="pinnacle",
            market_type="spread",
            captured_at=datetime.now(timezone.utc),
            odds_home=-110,
            line=10.0,
            source="api",
        )
        failures = run_validation_gate(payload, previous_line=-3.5, max_line_jump=15.0)
        assert failures == []

        failures = run_validation_gate(payload, previous_line=-3.5, max_line_jump=10.0)
        assert any("Line jump" in f.message for f in failures)


class TestAdaptiveParsing:
    def test_json_api_path_first(self):
        raw = {"markets": {"spread": {"line": -3.5}}}
        parser = AdaptiveParser(
            "line",
            [
                ParseStrategy("json_api", "1.0", parse_json_api_path(raw, ["markets", "spread", "line"])),
                ParseStrategy("regex", "1.0", parse_css_regex(r"line:\s*(-?\d+\.?\d*)", cast=parse_spread_line)),
            ],
        )
        result = parser.parse(raw)
        assert result.value == -3.5
        assert result.strategy_name == "json_api"

    def test_fallback_to_regex(self):
        html = '<div class="line">-7.5</div>'
        parser = AdaptiveParser(
            "line",
            [
                ParseStrategy(
                    "json_api",
                    "1.0",
                    lambda _: (_ for _ in ()).throw(KeyError("no json")),
                ),
                ParseStrategy(
                    "regex",
                    "1.0",
                    parse_css_regex(r"(-?\d+\.?\d*)", cast=parse_spread_line),
                ),
            ],
        )
        result = parser.parse(html)
        assert result.value == -7.5
        assert result.strategy_name == "regex"

    def test_american_odds_parsing(self):
        assert parse_american_odds("+150") == 150
        assert parse_american_odds("-110") == -110

    def test_all_strategies_fail_raises(self):
        parser = AdaptiveParser(
            "line",
            [ParseStrategy("bad", "1.0", lambda _: (_ for _ in ()).throw(ValueError("nope")))],
        )
        with pytest.raises(ValueError, match="All parse strategies failed"):
            parser.parse({})
