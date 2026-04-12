#!/usr/bin/env bash
# Run CFR pipeline with repo .venv so lifelines is available (Kaplan–Meier plots).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: expected $PY (create venv and pip install -r requirements with lifelines)" >&2
  exit 1
fi
exec "$PY" "$SCRIPT_DIR/run_cfr_analysis.py" "$@"
