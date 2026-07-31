#!/usr/bin/env bash
# Cost-scaled maze H2 wall-clock study: genetic vs genetic_filter @ injected sim delay.
# Descriptive only — does not amend F-B5 Holm families.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MAZE_EXPERIMENT_ROOT="${MAZE_EXPERIMENT_ROOT:-q1-v5-maze-cost-h2}"
export MAZE_PROPOSALS="${MAZE_PROPOSALS:-2500}"
export MAZE_SIM_COST_MS="${MAZE_SIM_COST_MS:-10}"

mkdir -p "$ROOT/artifacts/experiments/$MAZE_EXPERIMENT_ROOT"

for seed in $(seq 0 9); do
  ./scripts/run_experiment_batch.sh q1-v5-maze-genetic "$seed" "$seed"
  ./scripts/run_experiment_batch.sh q1-v5-maze-genetic-filter "$seed" "$seed"
done

"$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_maze_cost_h2.py" \
  --root "$ROOT/artifacts/experiments/$MAZE_EXPERIMENT_ROOT" \
  --sim-cost-ms "$MAZE_SIM_COST_MS"
