"""Shared paths and season helpers."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ODDS_DIR = DATA_DIR / "odds"
OUTPUT_DIR = ROOT / "output"
RESEARCH_DIR = ROOT / "research"
WEIGHTS_PATH = RESEARCH_DIR / "factor_weights.json"

load_dotenv(ROOT / ".env")

START_SEASON = int(os.getenv("NFL_EDGE_START_SEASON", "2006"))
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BOOK = os.getenv("ODDS_BOOK", "draftkings")


def current_nfl_season(today: date | None = None) -> int:
    """NFL season year: Sep–Dec = that year; Jan–Feb = previous calendar year."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def ensure_dirs() -> None:
    for path in (CACHE_DIR, ODDS_DIR, OUTPUT_DIR, RESEARCH_DIR):
        path.mkdir(parents=True, exist_ok=True)
