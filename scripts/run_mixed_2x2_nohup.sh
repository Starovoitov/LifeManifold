#!/usr/bin/env bash
# Launch parallel nohup workers for mixed-stack 2×2 re-run (archive_trace).
# Seeds round-robin: worker w runs seeds where (seed % MIXED_2X2_WORKERS == w).
# Default MIXED_2X2_WORKERS=2 (override: MIXED_2X2_WORKERS=4 ./scripts/run_mixed_2x2_nohup.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

EXP_ROOT="$ROOT/artifacts/experiments/q1-v3-mixed-2x2"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "QWEN_API_KEY is required for q1-v3-mixed-2x2 LLM arms" >&2
  exit 1
fi

export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
export LIFEMANIFOLD_LLM_PARALLEL_WORKERS="${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-4}"
export MIXED_2X2_WORKERS="${MIXED_2X2_WORKERS:-2}"
if [[ ! "$MIXED_2X2_WORKERS" =~ ^[0-9]+$ ]] || (( MIXED_2X2_WORKERS < 1 )); then
  echo "MIXED_2X2_WORKERS must be a positive integer (got: $MIXED_2X2_WORKERS)" >&2
  exit 1
fi

echo "=== mixed-2x2 nohup launch $(date -Is) ==="
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "MIXED_2X2_WORKERS=$MIXED_2X2_WORKERS"

for w in $(seq 0 $((MIXED_2X2_WORKERS - 1))); do
  log="$LOG_DIR/worker_${w}.log"
  pidfile="$LOG_DIR/worker_${w}.pid"
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Worker $w already running (pid $old_pid) — skip"
      continue
    fi
  fi
  nohup env MIXED_2X2_WORKERS="$MIXED_2X2_WORKERS" "$ROOT/scripts/run_mixed_2x2_worker.sh" "$w" >>"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "Started worker $w pid $(cat "$pidfile") log=$log seeds=$(for s in $(seq 0 9); do (( s % MIXED_2X2_WORKERS == w )) && echo -n "$s "; done)"
done

cat >"$LOG_DIR/README.txt" <<EOF
Mixed-stack 2×2 parallel workers (archive_trace tier: q1-v3-mixed-2x2)
MIXED_2X2_WORKERS=$MIXED_2X2_WORKERS

Monitor:
  tail -f $LOG_DIR/worker_*.log

PIDs:
  cat $LOG_DIR/worker_*.pid

When all workers finish, aggregate:
  $ROOT/.venv/bin/python $ROOT/scripts/aggregate_experiment_runs.py \\
    --root $EXP_ROOT --output $EXP_ROOT/summary.csv

Verify archive_trace on a completed run:
  ls $EXP_ROOT/hints/seed_0/archive_trace.jsonl
EOF

echo "=== launch complete $(date -Is) ==="
echo "See $LOG_DIR/README.txt"
