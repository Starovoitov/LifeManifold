#!/usr/bin/env bash
# Launch parallel nohup workers for Sphere RQ1 Phase B (stub/hints × minfit/uniform).
# Default SPHERE_FACTORIAL_WORKERS=2.
# Protocol: artifacts/Q1_RQ1_SPHERE_DOMAIN.md
# Do not launch until Phase A GO and a passing live preflight:
#   uv run python scripts/preflight_sphere_llm.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

EXP_ROOT="$ROOT/artifacts/experiments/q1-rq1-sphere-factorial"
LOG_DIR="$EXP_ROOT/logs"
CHECKPOINT="$ROOT/artifacts/surrogate/sphere_h1_mlp.joblib"
LLM_SPEC="$ROOT/worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml"
mkdir -p "$EXP_ROOT" "$LOG_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for q1-rq1-sphere-factorial (gpt-4o-mini-2024-07-18)" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing Sphere H1 surrogate checkpoint: $CHECKPOINT" >&2
  echo "Train: uv run python scripts/run_sphere_rq1.py train-surrogate --out $CHECKPOINT" >&2
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
export SPHERE_FACTORIAL_WORKERS="${SPHERE_FACTORIAL_WORKERS:-2}"
if [[ ! "$SPHERE_FACTORIAL_WORKERS" =~ ^[0-9]+$ ]] || (( SPHERE_FACTORIAL_WORKERS < 1 )); then
  echo "SPHERE_FACTORIAL_WORKERS must be a positive integer (got: $SPHERE_FACTORIAL_WORKERS)" >&2
  exit 1
fi

echo "=== sphere RQ1 Phase B nohup launch $(date -Is) ==="
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SPHERE_FACTORIAL_WORKERS=$SPHERE_FACTORIAL_WORKERS"
echo "LIFEMANIFOLD_LLM_PARALLEL_WORKERS=$LIFEMANIFOLD_LLM_PARALLEL_WORKERS"
echo "LLM_SPEC=$LLM_SPEC"

for w in $(seq 0 $((SPHERE_FACTORIAL_WORKERS - 1))); do
  log="$LOG_DIR/worker_${w}.log"
  pidfile="$LOG_DIR/worker_${w}.pid"
  if [[ -f "$pidfile" ]]; then
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Worker $w already running (pid $old_pid) — skip"
      continue
    fi
  fi
  nohup bash -c "
    set -euo pipefail
    cd \"$ROOT\"
    for seed in \$(seq 0 9); do
      if (( seed % $SPHERE_FACTORIAL_WORKERS == $w )); then
        echo \"[worker $w] seed=\$seed \$(date -Is)\"
        \"$ROOT/scripts/run_experiment_batch.sh\" q1-rq1-sphere-factorial \"\$seed\" \"\$seed\"
      fi
    done
    echo \"[worker $w] done \$(date -Is)\"
  " >>"$log" 2>&1 &
  echo $! >"$pidfile"
  seeds=""
  for s in $(seq 0 9); do
    if (( s % $SPHERE_FACTORIAL_WORKERS == $w )); then
      seeds+="$s "
    fi
  done
  echo "Started worker $w pid $(cat "$pidfile") log=$log seeds=$seeds"
done

cat >"$LOG_DIR/README.txt" <<EOF
Sphere RQ1 Phase B LLM 2×2 (tier: q1-rq1-sphere-factorial)
SPHERE_FACTORIAL_WORKERS=$SPHERE_FACTORIAL_WORKERS
LIFEMANIFOLD_LLM_PARALLEL_WORKERS=$LIFEMANIFOLD_LLM_PARALLEL_WORKERS
LLM_SPEC=$LLM_SPEC

Cells: llm_stub_minfit llm_stub_uniform llm_hints_minfit llm_hints_uniform
Budget: 5000 proposals, empty archive, dated gpt-4o-mini-2024-07-18
H1 leftover = live − stub at matched policy. Not Holm; not Sphere H2.

Monitor:
  tail -f $LOG_DIR/worker_*.log

Progress:
  $ROOT/.venv/bin/python $ROOT/scripts/analyze_rq1_sphere_factorial.py --root $EXP_ROOT
EOF

echo "=== launch complete $(date -Is) ==="
echo "See $LOG_DIR/README.txt"
