#!/usr/bin/env bash
# Cold-start mixed 2×2 worker: assigned seeds (seed % N_WORKERS == WORKER_ID) × 4 arms.
# Usage: COLD_SMOKE_WORKERS=2 COLD_SMOKE_SEED_START=0 COLD_SMOKE_SEED_END=4 \
#          ./scripts/run_cold_smoke_worker.sh WORKER_ID
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N_WORKERS="${COLD_SMOKE_WORKERS:-2}"
SEED_START="${COLD_SMOKE_SEED_START:-0}"
SEED_END="${COLD_SMOKE_SEED_END:-4}"
WORKER="${1:?worker id required}"
if [[ ! "$N_WORKERS" =~ ^[0-9]+$ ]] || (( N_WORKERS < 1 )); then
  echo "COLD_SMOKE_WORKERS must be a positive integer (got: $N_WORKERS)" >&2
  exit 1
fi
if [[ ! "$WORKER" =~ ^[0-9]+$ ]] || (( WORKER < 0 || WORKER >= N_WORKERS )); then
  echo "WORKER must be 0..$((N_WORKERS - 1)) (got: $WORKER, COLD_SMOKE_WORKERS=$N_WORKERS)" >&2
  exit 1
fi

export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
export LIFEMANIFOLD_LLM_PARALLEL_WORKERS="${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-2}"
export LIFEMANIFOLD_SKIP_EXPERIMENT_AGGREGATE=1

EXP_ROOT="$ROOT/artifacts/experiments/q1-v3-mixed-2x2-cold-smoke"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

echo "=== cold-smoke worker $WORKER start $(date -Is) ==="
echo "ROOT=$ROOT N_WORKERS=$N_WORKERS seeds ${SEED_START}-${SEED_END}:"
echo -n "  assigned: "
for s in $(seq "$SEED_START" "$SEED_END"); do
  (( s % N_WORKERS == WORKER )) && echo -n "$s "
done
echo

for seed in $(seq "$SEED_START" "$SEED_END"); do
  if (( seed % N_WORKERS != WORKER )); then
    continue
  fi
  echo "--- worker $WORKER seed $seed $(date -Is) ---"
  "$ROOT/scripts/run_experiment_batch.sh" q1-v3-mixed-2x2-cold-smoke "$seed" "$seed"
done

echo "=== cold-smoke worker $WORKER done $(date -Is) ==="
