#!/usr/bin/env bash
# Launch parallel nohup workers for H1 policy×hint missing cell (hints_minfit).
# Seeds round-robin: worker w runs seeds where (seed % WORKERS == w).
# Default POLICY_X_HINT_WORKERS=2 (override: POLICY_X_HINT_WORKERS=4 ./scripts/run_policy_x_hint_nohup.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

EXP_ROOT="$ROOT/artifacts/experiments/q1-h1-policy-x-hint"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

if [[ -z "${QWEN_API_KEY:-}" ]]; then
  echo "QWEN_API_KEY is required for q1-h1-policy-x-hint (qwen-turbo)" >&2
  exit 1
fi

export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
export LIFEMANIFOLD_LLM_PARALLEL_WORKERS="${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-4}"
export POLICY_X_HINT_WORKERS="${POLICY_X_HINT_WORKERS:-2}"
if [[ ! "$POLICY_X_HINT_WORKERS" =~ ^[0-9]+$ ]] || (( POLICY_X_HINT_WORKERS < 1 )); then
  echo "POLICY_X_HINT_WORKERS must be a positive integer (got: $POLICY_X_HINT_WORKERS)" >&2
  exit 1
fi

echo "=== policy×hint nohup launch $(date -Is) ==="
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "POLICY_X_HINT_WORKERS=$POLICY_X_HINT_WORKERS"

for w in $(seq 0 $((POLICY_X_HINT_WORKERS - 1))); do
  log="$LOG_DIR/worker_${w}.log"
  pidfile="$LOG_DIR/worker_${w}.pid"
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Worker $w already running (pid $old_pid) — skip"
      continue
    fi
  fi
  # shellcheck disable=SC2086
  nohup bash -c "
    set -euo pipefail
    cd \"$ROOT\"
    for seed in \$(seq 0 9); do
      if (( seed % $POLICY_X_HINT_WORKERS == $w )); then
        echo \"[worker $w] seed=\$seed \$(date -Is)\"
        \"$ROOT/scripts/run_experiment_batch.sh\" q1-h1-policy-x-hint \"\$seed\" \"\$seed\"
      fi
    done
    echo \"[worker $w] done \$(date -Is)\"
  " >>"$log" 2>&1 &
  echo $! >"$pidfile"
  seeds=""
  for s in $(seq 0 9); do
    if (( s % POLICY_X_HINT_WORKERS == w )); then
      seeds+="$s "
    fi
  done
  echo "Started worker $w pid $(cat "$pidfile") log=$log seeds=$seeds"
done

cat >"$LOG_DIR/README.txt" <<EOF
H1 policy×hint missing cell (tier: q1-h1-policy-x-hint → hints_minfit)
POLICY_X_HINT_WORKERS=$POLICY_X_HINT_WORKERS

Monitor:
  tail -f $LOG_DIR/worker_*.log

PIDs:
  cat $LOG_DIR/worker_*.pid

When all workers finish, aggregate + analyze:
  $ROOT/.venv/bin/python $ROOT/scripts/aggregate_experiment_runs.py \\
    --root $EXP_ROOT --output $EXP_ROOT/summary.csv
  $ROOT/.venv/bin/python $ROOT/scripts/analyze_h1_policy_x_hint.py

Protocol: $ROOT/artifacts/Q1_H1_POLICY_X_HINT.md
EOF

echo "=== launch complete $(date -Is) ==="
echo "See $LOG_DIR/README.txt"
