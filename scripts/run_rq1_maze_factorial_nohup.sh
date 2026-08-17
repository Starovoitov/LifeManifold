#!/usr/bin/env bash
# Launch parallel nohup workers for maze RQ1 Phase B (stub/hints × minfit/uniform).
# Seeds round-robin: worker w runs seeds where (seed % WORKERS == w).
# Default MAZE_FACTORIAL_WORKERS=2 (override: MAZE_FACTORIAL_WORKERS=4 ./scripts/run_rq1_maze_factorial_nohup.sh).
# Protocol: artifacts/Q1_RQ1_SECOND_DOMAIN.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

EXP_ROOT="$ROOT/artifacts/experiments/q1-rq1-maze-factorial"
LOG_DIR="$EXP_ROOT/logs"
CHECKPOINT="$ROOT/artifacts/surrogate/checkpoints/maze_v1.pkl"
LLM_SPEC="$ROOT/worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for q1-rq1-maze-factorial (gpt-4o-mini-2024-07-18)" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing maze surrogate checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$LLM_SPEC" ]]; then
  echo "Missing dated LLM spec: $LLM_SPEC" >&2
  exit 1
fi

export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
export LIFEMANIFOLD_LLM_CALL_LOG="${LIFEMANIFOLD_LLM_CALL_LOG:-1}"
export LIFEMANIFOLD_LLM_PARALLEL_WORKERS="${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-4}"
export LIFEMANIFOLD_SKIP_EXPERIMENT_AGGREGATE=1
export MAZE_FACTORIAL_WORKERS="${MAZE_FACTORIAL_WORKERS:-2}"
if [[ ! "$MAZE_FACTORIAL_WORKERS" =~ ^[0-9]+$ ]] || (( MAZE_FACTORIAL_WORKERS < 1 )); then
  echo "MAZE_FACTORIAL_WORKERS must be a positive integer (got: $MAZE_FACTORIAL_WORKERS)" >&2
  exit 1
fi

echo "=== maze RQ1 Phase B nohup launch $(date -Is) ==="
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "MAZE_FACTORIAL_WORKERS=$MAZE_FACTORIAL_WORKERS"
echo "LIFEMANIFOLD_LLM_PARALLEL_WORKERS=$LIFEMANIFOLD_LLM_PARALLEL_WORKERS"
echo "LLM_SPEC=$LLM_SPEC"

for w in $(seq 0 $((MAZE_FACTORIAL_WORKERS - 1))); do
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
      if (( seed % $MAZE_FACTORIAL_WORKERS == $w )); then
        echo \"[worker $w] seed=\$seed \$(date -Is)\"
        \"$ROOT/scripts/run_experiment_batch.sh\" q1-rq1-maze-factorial \"\$seed\" \"\$seed\"
      fi
    done
    echo \"[worker $w] done \$(date -Is)\"
  " >>"$log" 2>&1 &
  echo $! >"$pidfile"
  seeds=""
  for s in $(seq 0 9); do
    if (( s % MAZE_FACTORIAL_WORKERS == w )); then
      seeds+="$s "
    fi
  done
  echo "Started worker $w pid $(cat "$pidfile") log=$log seeds=$seeds"
done

cat >"$LOG_DIR/README.txt" <<EOF
Maze RQ1 Phase B LLM 2×2 (tier: q1-rq1-maze-factorial)
MAZE_FACTORIAL_WORKERS=$MAZE_FACTORIAL_WORKERS
LIFEMANIFOLD_LLM_PARALLEL_WORKERS=$LIFEMANIFOLD_LLM_PARALLEL_WORKERS
LLM_SPEC=$LLM_SPEC

Cells: llm_stub_minfit llm_stub_uniform llm_hints_minfit llm_hints_uniform
Budget: 5000 proposals, empty archive, dated gpt-4o-mini-2024-07-18
Not Holm; not H5 llm_stub/llm_hints @ 32.5k.

Monitor:
  tail -f $LOG_DIR/worker_*.log

PIDs:
  cat $LOG_DIR/worker_*.pid

Progress:
  $ROOT/.venv/bin/python $ROOT/scripts/analyze_rq1_maze_factorial.py --root $EXP_ROOT

When all workers finish, aggregate:
  $ROOT/.venv/bin/python $ROOT/scripts/aggregate_experiment_runs.py \\
    --root $EXP_ROOT --output $EXP_ROOT/summary.csv
EOF

echo "=== launch complete $(date -Is) ==="
echo "See $LOG_DIR/README.txt"
