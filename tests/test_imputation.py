from __future__ import annotations

from sports_ev.models.imputation import blend_with_market, clip_probability, impute_features


class TestImputation:
    def test_impute_missing_features(self):
        imputations = {"a": 1.0, "b": 2.0}
        features = {"a": 3.0, "b": None}
        imputed, missing_ratio = impute_features(features, ["a", "b"], imputations)
        assert imputed["a"] == 3.0
        assert imputed["b"] == 2.0
        assert missing_ratio == 0.5

    def test_clip_probability(self):
        assert clip_probability(0.99) == 0.90
        assert clip_probability(0.01) == 0.10

    def test_blend_with_market_pulls_extreme_toward_market(self):
        blended = blend_with_market(0.98, 0.60, missing_ratio=0.4)
        assert blended < 0.98
        assert blended > 0.60
