#!/usr/bin/env bash
# Aggregate q1-v3-mixed-2x2 after all parallel workers finish.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXP_ROOT="$ROOT/artifacts/experiments/q1-v3-mixed-2x2"
"$ROOT/.venv/bin/python" "$ROOT/scripts/aggregate_experiment_runs.py" \
  --root "$EXP_ROOT" \
  --output "$EXP_ROOT/summary.csv"
echo "Wrote $EXP_ROOT/summary.csv"
