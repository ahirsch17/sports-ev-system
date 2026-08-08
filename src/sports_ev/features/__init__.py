"""Versioned NFL feature pipeline with leakage guards."""

FEATURE_VERSION = "nfl_spread_v1"

from sports_ev.features.leakage import LeakageError, assert_no_leakage
from sports_ev.features.pipeline import FeaturePipeline, FeatureVector
from sports_ev.features.version import feature_version

__all__ = [
    "FEATURE_VERSION",
    "FeaturePipeline",
    "FeatureVector",
    "LeakageError",
    "assert_no_leakage",
    "feature_version",
]
