#!/usr/bin/env bash
# Mixed-stack 2×2 worker: runs assigned seeds (seed % N_WORKERS == WORKER_ID) × 4 arms.
# Usage: MIXED_2X2_WORKERS=2 ./scripts/run_mixed_2x2_worker.sh WORKER_ID
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N_WORKERS="${MIXED_2X2_WORKERS:-4}"
WORKER="${1:?worker id required}"
if [[ ! "$N_WORKERS" =~ ^[0-9]+$ ]] || (( N_WORKERS < 1 )); then
  echo "MIXED_2X2_WORKERS must be a positive integer (got: $N_WORKERS)" >&2
  exit 1
fi
if [[ ! "$WORKER" =~ ^[0-9]+$ ]] || (( WORKER < 0 || WORKER >= N_WORKERS )); then
  echo "WORKER must be 0..$((N_WORKERS - 1)) (got: $WORKER, MIXED_2X2_WORKERS=$N_WORKERS)" >&2
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
echo "N_WORKERS=$N_WORKERS seeds: $(for s in $(seq 0 9); do (( s % N_WORKERS == WORKER )) && echo -n "$s "; done)"

for seed in $(seq 0 9); do
  if (( seed % N_WORKERS != WORKER )); then
    continue
  fi
  echo "--- worker $WORKER seed $seed $(date -Is) ---"
  "$ROOT/scripts/run_experiment_batch.sh" q1-v3-mixed-2x2 "$seed" "$seed"
done

echo "=== mixed-2x2 worker $WORKER done $(date -Is) ==="
