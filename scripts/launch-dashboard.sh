#!/usr/bin/env bash
# macOS/Linux counterpart of launch-dashboard.bat.
set -euo pipefail
cd "$(dirname "$0")/.."

"$(dirname "$0")/ensure-docker-postgres.sh"

echo
echo "Launching SportsPredictor EV Control Room..."
echo "Close the browser tab to stop the server, or press Ctrl+C here."
echo
exec .venv/bin/python -u -m sports_ev.dashboard
