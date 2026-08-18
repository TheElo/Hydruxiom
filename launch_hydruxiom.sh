#!/usr/bin/env bash
# Linux launcher for Hydruxiom - 3D Tag Space Explorer
# Equivalent to launch_hydruxiom.bat on Windows.
#
# Usage:
#   ./launch_hydruxiom.sh          (after: chmod +x launch_hydruxiom.sh)
#   bash launch_hydruxiom.sh       (no chmod needed)

set -e
cd "$(dirname "$0")"

echo "Launching Hydruxiom - 3D Tag Space Explorer..."

# Create venv on first run
if [ ! -f ".venv/bin/python" ]; then
    echo "First run: creating virtual environment..."
    python3 -m venv .venv
fi

# Install deps (quiet; re-runs are fast when nothing changed)
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt

exec .venv/bin/python main.py
