#!/usr/bin/env bash
# Run one condition × seed for the Q1 experiment matrix.
# Usage: ./scripts/run_experiment_batch.sh pilot|q1-min|q1-full|q1-full-filter|shadow [first_seed] [last_seed]
#
# q1-full-filter: filter arm only under artifacts/experiments/q1-full/; requires
# stub + hints nightly_run_summary.json for each seed (e.g. after cp from q1-min).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIER="${1:-pilot}"
SEED_START="${2:-0}"
SEED_END="${3:-$SEED_START}"
FILTER_ONLY=false
case "$TIER" in
  q1-full-filter)
    TIER=q1-full
    FILTER_ONLY=true
    ;;
esac

EXP_ROOT="$ROOT/artifacts/experiments"
BASELINE_ARCHIVE="$ROOT/artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
TRAIN_SCRIPT="$ROOT/scripts/train_surrogate.py"
RUN_SCRIPT="$ROOT/scripts/run_github_llm_map_elites.py"
AGG_SCRIPT="$ROOT/scripts/aggregate_experiment_runs.py"

SCHEDULER_STUB_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub.yaml"
SCHEDULER_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm.yaml"
SCHEDULER_FILTER_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter.yaml"
SCHEDULER_SHADOW_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_hints.yaml"
SCHEDULER_SHADOW_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow.yaml"
SCHEDULER_STUB_PILOT="$ROOT/worldspace/specs/map_elites_scheduler_github_llm_stub.yaml"
SCHEDULER_HINTS_PILOT="$ROOT/worldspace/specs/map_elites_scheduler_github_llm.yaml"

case "$TIER" in
  pilot)
    ITERATIONS=120
    EXP_DIR="$EXP_ROOT/pilot"
    SCHEDULER_STUB="$SCHEDULER_STUB_PILOT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_PILOT"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-min)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-min"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-full)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-full"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    SCHEDULER_FILTER="$SCHEDULER_FILTER_NIGHTLY"
    RUN_FILTER=true
    RUN_SHADOW=false
    ;;
  shadow)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/shadow"
    SCHEDULER_HINTS="$SCHEDULER_SHADOW_HINTS_NIGHTLY"
    SCHEDULER_FILTER="$SCHEDULER_SHADOW_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=true
    ;;
  *)
    echo "Unknown tier: $TIER (use pilot|q1-min|q1-full|q1-full-filter|shadow)" >&2
    exit 1
    ;;
esac

# Long nightly tiers: cap LLM HTTP concurrency and log per-iteration emit/eval timing.
# Override via LIFEMANIFOLD_LLM_PARALLEL_WORKERS / LIFEMANIFOLD_LOG_ITERATION_TIMING.
apply_long_run_llm_defaults() {
  if [[ -z "${LIFEMANIFOLD_LOG_ITERATION_TIMING:-}" ]]; then
    export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
  fi
  if [[ -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
    export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=4
  fi
}

case "$TIER" in
  q1-min|q1-full|shadow) apply_long_run_llm_defaults ;;
esac

# Filter-only tier: lower LLM HTTP concurrency (burst resets under parallel load).
if [[ "$FILTER_ONLY" == true && -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
  export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2
fi

if [[ ! -f "$BASELINE_ARCHIVE" ]]; then
  echo "Missing baseline archive: $BASELINE_ARCHIVE" >&2
  echo "Run: uv run python -m worldspace.scripts.run_map_elites_nightly" >&2
  exit 1
fi

CHECKPOINT="$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training surrogate checkpoint..."
  uv run python "$TRAIN_SCRIPT" \
    --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
    --checkpoint-path "$CHECKPOINT" \
    --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json" \
    --mlp-dropout-p 0.05 \
    --mlp-uncertainty-method ensemble_mc \
    --mlp-mc-samples 16 \
    --no-quality-gate
fi

CALIBRATION="$ROOT/artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl"
if [[ ("$RUN_FILTER" == true || "$RUN_SHADOW" == true) && ! -f "$CALIBRATION" ]]; then
  echo "Training uncertainty calibration (required for filter arm)..."
  uv run python "$TRAIN_SCRIPT" \
    --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
    --checkpoint-path "$CHECKPOINT" \
    --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json" \
    --calibrate \
    --calibration-path "$CALIBRATION" \
    --no-quality-gate
fi

mkdir -p "$EXP_DIR"

require_stub_hints_for_seed() {
  local seed="$1"
  local stub_summary="$EXP_DIR/stub/seed_${seed}/nightly_run_summary.json"
  local hints_summary="$EXP_DIR/hints/seed_${seed}/nightly_run_summary.json"
  if [[ ! -f "$stub_summary" || ! -f "$hints_summary" ]]; then
    echo "filter-only: missing completed stub/hints for seed $seed" >&2
    echo "  expected: $stub_summary" >&2
    echo "  expected: $hints_summary" >&2
    exit 1
  fi
}

remove_incomplete_run_dir() {
  local out="$1"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    return 0
  fi
  if [[ -d "$out" ]] && [[ -n "$(ls -A "$out" 2>/dev/null || true)" ]]; then
    echo "Removing incomplete run artifacts: $out" >&2
    rm -rf "$out"
  fi
}

run_one() {
  local condition="$1"
  local scheduler="$2"
  local seed="$3"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ "$condition" == "hints" || "$condition" == "filter" ]]; then
    extra+=(--require-surrogate-quality-gate)
  fi
  echo "=== tier=$TIER condition=$condition seed=$seed ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$RUN_SCRIPT" \
      --scheduler "$scheduler" \
      --output-dir "$out" \
      --seed "$seed" \
      --iterations "$ITERATIONS" \
      --load-archive "$BASELINE_ARCHIVE" \
      --llm-provider qwen \
      "${extra[@]}"
  ) 9>"$lock_file"
}

for seed in $(seq "$SEED_START" "$SEED_END"); do
  if [[ "$RUN_SHADOW" == true ]]; then
    run_one hints "$SCHEDULER_HINTS" "$seed"
    # Protocol layout: shadow-mode run under filter/ (archive must match hints).
    run_one filter "$SCHEDULER_FILTER" "$seed"
  else
    if [[ "$FILTER_ONLY" == true ]]; then
      require_stub_hints_for_seed "$seed"
    else
      run_one stub "$SCHEDULER_STUB" "$seed"
      run_one hints "$SCHEDULER_HINTS" "$seed"
    fi
    if [[ "$RUN_FILTER" == true ]]; then
      run_one filter "$SCHEDULER_FILTER" "$seed"
    fi
  fi
done

if [[ -f "$AGG_SCRIPT" ]]; then
  if ! uv run python "$AGG_SCRIPT" --root "$EXP_DIR" --output "$EXP_DIR/summary.csv"; then
    echo "WARNING: failed to write $EXP_DIR/summary.csv (runs may still be valid)" >&2
    exit 1
  fi
  echo "Wrote $EXP_DIR/summary.csv"
fi
