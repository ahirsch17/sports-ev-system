#!/usr/bin/env bash
# macOS/Linux counterpart of scheduled-refresh.bat. Example crontab (every 4 hours):
#   0 */4 * * * /path/to/sports-ev-system/scripts/scheduled-refresh.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/scheduled-refresh.log
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

{
  echo
  echo "============================================================"
  echo "[$(stamp)] scheduled-refresh start"
} >> "$LOG"

if ! scripts/ensure-docker-postgres.sh >> "$LOG" 2>&1; then
  echo "[$(stamp)] ABORT: Docker/Postgres not ready" >> "$LOG"
  exit 1
fi

echo "[$(stamp)] refresh-cycle..." >> "$LOG"
.venv/bin/python -m sports_ev.cli refresh-cycle --all-sports --days 3 --book draftkings \
  --sync-probables --capture-closing --score-lookback-days 14 >> "$LOG" 2>&1
RC=$?

echo "[$(stamp)] paper-report..." >> "$LOG"
.venv/bin/python -m sports_ev.cli paper-report --days 7 >> "$LOG" 2>&1

echo "[$(stamp)] scheduled-refresh done exit=$RC" >> "$LOG"
exit $RC
