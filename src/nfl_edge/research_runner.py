"""Package-importable research runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def run_research(start_season: int = 2006, end_season: int | None = None):
    path = Path(__file__).resolve().parents[2] / "research" / "run_factor_research.py"
    spec = importlib.util.spec_from_file_location("run_factor_research", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_research(start_season=start_season, end_season=end_season)
