#!/usr/bin/env bash
# Launch the Mac speaker detection app.
# Creates a venv on first run and installs dependencies.
set -e

cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ ! -f ".deps_installed" ]; then
  echo "==> Installing dependencies (this may take a few minutes)..."
  pip install --upgrade pip
  pip install -r requirements.txt
  touch .deps_installed
fi

echo "==> Launching Speaker Detect..."
exec python speaker_detect.py
