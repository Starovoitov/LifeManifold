#!/usr/bin/env bash
# Tier 2: CPU full budget @ 32.5k proposals (genetic + genetic_filter, seeds 0-9).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export DUNGEON_EXPERIMENT_ROOT=q1-v4-dungeon-full-cpu
export DUNGEON_PROPOSALS=32500

LOG="$ROOT/artifacts/experiments/q1-v4-dungeon-full-cpu/tier2_run.log"
mkdir -p "$(dirname "$LOG")"

exec >>"$LOG" 2>&1

echo "=== Tier 2 start $(date -Is) ==="
echo "ROOT=$ROOT"
echo "DUNGEON_EXPERIMENT_ROOT=$DUNGEON_EXPERIMENT_ROOT"
echo "DUNGEON_PROPOSALS=$DUNGEON_PROPOSALS"

./scripts/run_experiment_batch.sh q1-v4-dungeon-genetic 0 9
./scripts/run_experiment_batch.sh q1-v4-dungeon-genetic-filter 0 9

echo "=== Aggregating $(date -Is) ==="
.venv/bin/python scripts/aggregate_experiment_runs.py \
  --root "artifacts/experiments/$DUNGEON_EXPERIMENT_ROOT"

echo "=== Stats v4-dungeon-cpu-full $(date -Is) ==="
.venv/bin/python scripts/analyze_q1_statistics.py \
  --family v4-dungeon-cpu-full \
  --dungeon-root "artifacts/experiments/$DUNGEON_EXPERIMENT_ROOT"

echo "=== Tier 2 done $(date -Is) ==="
