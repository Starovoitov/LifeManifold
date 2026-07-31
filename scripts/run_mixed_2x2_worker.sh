#!/usr/bin/env bash
# Mixed-stack 2×2 worker: runs assigned seeds (seed % 4 == WORKER_ID) × 4 arms.
# Usage: ./scripts/run_mixed_2x2_worker.sh WORKER_ID   # WORKER_ID ∈ {0,1,2,3}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKER="${1:?worker id 0-3 required}"
if [[ ! "$WORKER" =~ ^[0-3]$ ]]; then
  echo "WORKER must be 0, 1, 2, or 3 (got: $WORKER)" >&2
  exit 1
fi

export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
export LIFEMANIFOLD_LLM_PARALLEL_WORKERS="${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-4}"
export LIFEMANIFOLD_SKIP_EXPERIMENT_AGGREGATE=1

EXP_ROOT="$ROOT/artifacts/experiments/q1-v3-mixed-2x2"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

echo "=== mixed-2x2 worker $WORKER start $(date -Is) ==="
echo "ROOT=$ROOT"
echo "seeds: $(for s in $(seq 0 9); do (( s % 4 == WORKER )) && echo -n "$s "; done)"

for seed in $(seq 0 9); do
  if (( seed % 4 != WORKER )); then
    continue
  fi
  echo "--- worker $WORKER seed $seed $(date -Is) ---"
  "$ROOT/scripts/run_experiment_batch.sh" q1-v3-mixed-2x2 "$seed" "$seed"
done

echo "=== mixed-2x2 worker $WORKER done $(date -Is) ==="
