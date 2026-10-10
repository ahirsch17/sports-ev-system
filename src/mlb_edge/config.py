"""MLB paths and season helpers."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "mlb"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "output" / "mlb"
RESEARCH_DIR = ROOT / "research" / "mlb"
WEIGHTS_PATH = RESEARCH_DIR / "factor_weights.json"

load_dotenv(ROOT / ".env")


def current_mlb_season(today: date | None = None) -> int:
    today = today or date.today()
    # MLB season year: Mar–Dec = that year; Jan–Feb = previous
    return today.year if today.month >= 3 else today.year - 1


def ensure_dirs() -> None:
    for p in (CACHE_DIR, OUTPUT_DIR, RESEARCH_DIR):
        p.mkdir(parents=True, exist_ok=True)
