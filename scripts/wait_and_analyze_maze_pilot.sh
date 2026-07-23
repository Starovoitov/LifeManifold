#!/usr/bin/env bash
# Wait for five-arm maze pilot completion, then aggregate + write ANALYSIS.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PILOT_ROOT="${1:-$ROOT/artifacts/experiments/q1-v5-maze-pilot}"
LOG="$PILOT_ROOT/logs/wait_analyze.log"
mkdir -p "$(dirname "$LOG")"

echo "$(date -Is) waiting for 5/5 summaries in $PILOT_ROOT" | tee -a "$LOG"
while true; do
  n="$(find "$PILOT_ROOT" -name nightly_run_summary.json 2>/dev/null | wc -l)"
  echo "$(date -Is) complete=$n/5" | tee -a "$LOG"
  if [[ "$n" -ge 5 ]]; then
    break
  fi
  sleep 120
done

cd "$ROOT"
uv run python scripts/aggregate_experiment_runs.py \
  --root "$PILOT_ROOT" \
  --output "$PILOT_ROOT/summary.csv" 2>&1 | tee -a "$LOG"
uv run python scripts/analyze_maze_pilot.py --root "$PILOT_ROOT" 2>&1 | tee -a "$LOG"
echo "$(date -Is) DONE analysis" | tee -a "$LOG"
