#!/usr/bin/env bash
# Launch cold-start mixed 2×2 (empty archive).
# Usage:
#   ./scripts/run_cold_smoke_nohup.sh [first_seed] [last_seed]
#   COLD_SMOKE_WORKERS=2 ./scripts/run_cold_smoke_nohup.sh 0 4
# Default: single process. With COLD_SMOKE_WORKERS>1: round-robin seed workers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

SEED_START="${1:-0}"
SEED_END="${2:-0}"
EXP_ROOT="$ROOT/artifacts/experiments/q1-v3-mixed-2x2-cold-smoke"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "QWEN_API_KEY is required for q1-v3-mixed-2x2-cold-smoke" >&2
  exit 1
fi

export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
export LIFEMANIFOLD_LLM_PARALLEL_WORKERS="${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-2}"
export COLD_SMOKE_WORKERS="${COLD_SMOKE_WORKERS:-1}"
export COLD_SMOKE_SEED_START="$SEED_START"
export COLD_SMOKE_SEED_END="$SEED_END"

if [[ ! "$COLD_SMOKE_WORKERS" =~ ^[0-9]+$ ]] || (( COLD_SMOKE_WORKERS < 1 )); then
  echo "COLD_SMOKE_WORKERS must be a positive integer (got: $COLD_SMOKE_WORKERS)" >&2
  exit 1
fi

echo "=== cold-smoke nohup launch $(date -Is) ==="
echo "EXP_ROOT=$EXP_ROOT seeds=${SEED_START}-${SEED_END} COLD_SMOKE_WORKERS=$COLD_SMOKE_WORKERS"
echo "LIFEMANIFOLD_LLM_PARALLEL_WORKERS=$LIFEMANIFOLD_LLM_PARALLEL_WORKERS"

# Namespace pid/log by seed range so wave B (5–9) can run beside wave A (0–4).
WAVE_TAG="${SEED_START}_${SEED_END}"

if (( COLD_SMOKE_WORKERS == 1 )); then
  log="$LOG_DIR/cold_smoke_${WAVE_TAG}.log"
  pidfile="$LOG_DIR/cold_smoke_${WAVE_TAG}.pid"
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Cold smoke already running (pid $old_pid) — skip" >&2
      exit 0
    fi
  fi
  nohup "$ROOT/scripts/run_experiment_batch.sh" q1-v3-mixed-2x2-cold-smoke "$SEED_START" "$SEED_END" \
    >>"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "Started single pid $(cat "$pidfile") log=$log"
else
  for w in $(seq 0 $((COLD_SMOKE_WORKERS - 1))); do
    log="$LOG_DIR/worker_${WAVE_TAG}_${w}.log"
    pidfile="$LOG_DIR/worker_${WAVE_TAG}_${w}.pid"
    if [[ -f "$pidfile" ]]; then
      old_pid="$(cat "$pidfile" 2>/dev/null || true)"
      if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Worker $w already running (pid $old_pid) — skip"
        continue
      fi
    fi
    nohup env COLD_SMOKE_WORKERS="$COLD_SMOKE_WORKERS" \
      COLD_SMOKE_SEED_START="$SEED_START" \
      COLD_SMOKE_SEED_END="$SEED_END" \
      LIFEMANIFOLD_LLM_PARALLEL_WORKERS="$LIFEMANIFOLD_LLM_PARALLEL_WORKERS" \
      "$ROOT/scripts/run_cold_smoke_worker.sh" "$w" >>"$log" 2>&1 &
    echo $! >"$pidfile"
    echo "Started worker $w pid $(cat "$pidfile") log=$log seeds=$(for s in $(seq "$SEED_START" "$SEED_END"); do (( s % COLD_SMOKE_WORKERS == w )) && echo -n "$s "; done)"
  done
fi

cat >"$LOG_DIR/README.txt" <<EOF
Cold-start mixed 2×2 (tier q1-v3-mixed-2x2-cold-smoke)
Seeds: ${SEED_START}-${SEED_END}
COLD_SMOKE_WORKERS=$COLD_SMOKE_WORKERS
Lock: artifacts/Q1_COLD_START_SMOKE.md

Monitor:
  tail -f $LOG_DIR/worker_*.log
  # or: tail -f $LOG_DIR/cold_smoke.log

When finished, aggregate:
  $ROOT/.venv/bin/python $ROOT/scripts/aggregate_experiment_runs.py \\
    --root $EXP_ROOT --output $EXP_ROOT/summary.csv
EOF

echo "=== launch complete $(date -Is) ==="
echo "See $LOG_DIR/README.txt"
