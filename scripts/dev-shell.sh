#!/usr/bin/env bash
# Source this once per Git Bash session, or add to ~/.bashrc:
#   source ~/Projects/github/ahirsch17/sports-ev-system/scripts/docker-path.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=docker-path.sh
source "$SCRIPT_DIR/docker-path.sh"

PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

if [ -d ".venv/Scripts" ]; then
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
fi

echo "Docker: $(command -v docker)"
echo "Project: $PROJECT_DIR"
