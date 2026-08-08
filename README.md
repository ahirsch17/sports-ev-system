# SportsPredictor (sports-ev-system)

**Canonical repo** for SportsPredictor: multi-sport +EV research, paper betting, and CLV tracking.  
The old standalone [`sportsPredictor`](https://github.com/ahirsch17/sportsPredictor) FastAPI demo is retired — all active code lives here.

## What this is

- **NFL + MLB** models (spread / moneyline), Pinnacle + DraftKings + FanDuel ingestion
- **Postgres** storage, circuit breakers, scheduled refresh
- **Streamlit control room** with the original SportsPredictor dark UI (hero, grid, brand)
- **Paper bets only** — no automatic real-money wagering

## Quick start (Windows)

```bat
scripts\launch-dashboard.bat
```

Starts Docker/Postgres if needed, then opens the dashboard at `http://localhost:8501`.

## Automation

```powershell
.\scripts\install-scheduled-task.ps1
```

Runs `scheduled-refresh.bat` every 8 hours (odds, +EV picks, settle). Logs: `logs/scheduled-refresh.log`.

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

## Repo layout

- `src/sports_ev/` — application code
- `scripts/` — launchers and Task Scheduler installer
- `docker-compose.yml` — Postgres on port **5433**
