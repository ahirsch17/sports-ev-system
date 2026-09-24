# SportsPredictor (sports-ev-system)

Multi-sport +EV research toolkit with paper betting and CLV tracking.

This is the canonical codebase. The old standalone FastAPI demo repo is gone.

## Features

- NFL and MLB models (spread / moneyline)
- Odds ingestion from Pinnacle, DraftKings, and FanDuel
- Postgres storage, circuit breakers, scheduled refresh
- Streamlit control room UI
- Paper bets only (no automatic real-money wagering)

## Quick start (Windows)

```bat
scripts\launch-dashboard.bat
```

Starts Docker/Postgres if needed, then opens `http://localhost:8501`.

## Automation

```powershell
.\scripts\install-scheduled-task.ps1
```

Runs `scheduled-refresh.bat` every 4 hours. Logs: `logs/scheduled-refresh.log`.

## CLI

```bash
pip install -e ".[dev,dashboard]"
docker compose up -d
python -m sports_ev.cli init-db
python -m sports_ev.cli migrate-db
python -m sports_ev.cli paper-report --days 7
python -m sports_ev.cli refresh-cycle --all-sports --book draftkings --sync-probables --capture-closing
```

## Dev

```bash
pytest
python -m sports_ev.cli model-scorecard --sport mlb
```

## Layout

- `src/sports_ev/`: application code
- `scripts/`: launchers and Task Scheduler installer
- `docker-compose.yml`: Postgres on port 5433
