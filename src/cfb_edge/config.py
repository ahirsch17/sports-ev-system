"""CFB paths and season helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "cfb"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "output" / "cfb"
RESEARCH_DIR = ROOT / "research" / "cfb"
WEIGHTS_PATH = RESEARCH_DIR / "factor_weights.json"

load_dotenv(ROOT / ".env")


def current_cfb_season(today: date | None = None) -> int:
    """CFB season year: Aug–Dec = that year; Jan–Jul = previous (bowl/carryover)."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def ensure_dirs() -> None:
    for p in (CACHE_DIR, OUTPUT_DIR, RESEARCH_DIR):
        p.mkdir(parents=True, exist_ok=True)
