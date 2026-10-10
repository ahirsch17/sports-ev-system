# SportsPredictor (sports-ev-system)

Multi-sport +EV research toolkit: fundamental models for NFL, MLB and college football,
priced against a sharp book (Pinnacle) and soft books (DraftKings, FanDuel), with paper
betting, Kelly sizing and closing-line-value (CLV) tracking.

Paper bets only. Nothing here places real-money wagers. Edges are small and noisy.

## Two halves, one repo

| | `sports_ev` (pipeline) | `nfl_edge` / `mlb_edge` / `cfb_edge` (models + board) |
| --- | --- | --- |
| Data | Postgres; Pinnacle guest API, DK/FD scrapers, ESPN scores | nflverse parquet caches, MLB Stats API, cfbfastR / CFBD |
| Models | LightGBM NFL spread + MLB moneyline, leakage-audited features | Formula models: Elo + points, HFA eras, weather, form, injuries x depth charts, deep matchups, situational; props |
| Pricing | Devig (multiplicative / Shin), EV, fractional Kelly | Model fair prices vs DK (The Odds API) or schedule lines |
| Tracking | Paper bets with audit trail, CLV settlement, walk-forward backtests | JSON bet log (`output/bet_log.json`) |
| UI | Streamlit control room (`:8501`) | FastAPI board (`:8510`) |

**The bridge:** `sports_ev.models.nfl_edge_strategy.NflEdgeSpreadStrategy` runs the full
`nfl_edge` model inside the sports_ev pipeline. It blends the nfl_edge margin with the
margin implied by Pinnacle's devigged spread, converts to a cover probability at the
soft book's line, and flows through Kelly sizing, paper bets, CLV settlement and backtests:

```bash
sports-ev flag-opportunities --sport nfl --strategy nfl-edge --book draftkings
sports-ev run-backtest --sport nfl --strategy nfl-edge --season 2025
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,dashboard]"
cp .env.example .env                 # optional: ODDS_API_KEY for live DK prices on the board
docker compose up -d                 # Postgres on port 5433
sports-ev init-db && sports-ev migrate-db
```

## Daily use

```bash
# Pipeline: fixtures, odds, scores, flags, settlement, closing lines
sports-ev refresh-cycle --all-sports --book draftkings --sync-probables --capture-closing
sports-ev paper-report --days 7

# Dashboards
scripts/launch-dashboard.sh          # Streamlit control room (Windows: scripts\launch-dashboard.bat)
sports-edge ui                       # NFL / MLB / CFB board at http://127.0.0.1:8510

# Model views
sports-edge predict --week 6         # NFL spreads, totals, team yards
sports-edge edges --week 6 --props   # model vs DK (needs ODDS_API_KEY; else schedule lines)
sports-edge mlb --days 2
sports-edge cfb
```

Scheduled refresh: `scripts/install-scheduled-task.ps1` on Windows, or cron on macOS/Linux
(see the header of `scripts/scheduled-refresh.sh`). Both run every 4 hours and log to
`logs/scheduled-refresh.log`.

## Research and what the numbers say

Weights live in `research/factor_weights.json` (NFL), `research/mlb/` and `research/cfb/`.

| Script | Refits |
| --- | --- |
| `sports-edge research` | HFA by era, weather, momentum (merges into existing weights) |
| `research/audit_factor_ablation.py` | Which NFL factors earn their weight |
| `research/audit_iterate_confidence.py` | Confidence gates, props, injuries |
| `research/audit_cfb.py` | CFB Elo / form configuration |
| `research/fit_market_blend.py` | How much the nfl_edge margin should move the sharp line |

Walk-forward findings:

- **NFL winners:** about 64% straight up, about 71% when the model margin is 5+ points.
- **NFL against the spread:** about 51% on its own, which is below the 52.4% needed at -110.
  `fit_market_blend.py` (1,871 games, 2019-2025) puts the best weight on the model margin
  at about 0.15 versus the closing market, a small but consistent log-loss improvement. So
  the bridge treats nfl_edge as a tilt on Pinnacle's number, not a replacement, and only
  flags bets when that tilt plus any soft-book line or price gap clears `MIN_EDGE_PCT`.
- **MLB:** about 56% overall; about 65% when both starters are known with an ERA gap of 1+.
- **CFB:** about 70% both-FBS; gated big mismatches are mostly chalk, so prefer spreads.
- **Parlays:** the board only builds same-game tickets from moneyline legs (spreads and
  totals lost to the juice in audits), so it usually shows none. That is intentional.

## Conventions

- Spread lines in `odds_snapshots` and strategy candidates are from the **home** team's
  perspective (home -3.5).
- Paper bets store the **bettor's own** line (away +3.5). Settlement and CLV use that
  side's closing number, and positive CLV means you beat the close.
- nflverse `spread_line` is the expected home margin, so the home line is its negation;
  nflverse kickoff times are US Eastern.

## Dev

```bash
pytest                                       # 175 tests; runner tests take ~40s
sports-ev model-scorecard --sport mlb
```

## Layout

- `src/sports_ev/`: pipeline (db, odds scrapers, pricing, features, models, paper, backtest, dashboard)
- `src/nfl_edge/`, `src/mlb_edge/`, `src/cfb_edge/`: formula models, ingest, edge/advice, tracker
- `research/`: factor studies and weight files
- `ui/`: FastAPI + static board
- `scripts/`: launchers and schedulers (Windows `.bat`/`.ps1`, macOS/Linux `.sh`)
- `docker-compose.yml`: Postgres on port 5433
- `data/`, `output/`, `logs/`, `artifacts/`: local caches and outputs (gitignored)
