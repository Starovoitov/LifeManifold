#!/usr/bin/env bash
# Run one condition × seed for the Q1 experiment matrix.
# Usage: ./scripts/run_experiment_batch.sh TIER [first_seed] [last_seed]
#
# Grid tiers:  pilot | q1-min | q1-full | q1-full-filter | shadow
# CVT tiers:   q1-cvt-min | q1-cvt | q1-cvt-filter | cvt-shadow
#
# q1-*-filter: filter arm only; requires completed stub + hints for each seed.
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
  q1-cvt-filter)
    TIER=q1-cvt
    FILTER_ONLY=true
    ;;
esac

EXP_ROOT="$ROOT/artifacts/experiments"
GRID_BASELINE_ARCHIVE="$ROOT/artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
CVT_BASELINE_ARCHIVE="$ROOT/artifacts/map_elites_nightly/cvt/baseline/map_elites_archive.jsonl"
BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
ARCHIVE_TYPE=grid

TRAIN_SCRIPT="$ROOT/scripts/train_surrogate.py"
RUN_SCRIPT="$ROOT/scripts/run_github_llm_map_elites.py"
AGG_SCRIPT="$ROOT/scripts/aggregate_experiment_runs.py"

SCHEDULER_STUB_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub.yaml"
SCHEDULER_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm.yaml"
SCHEDULER_FILTER_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter.yaml"
SCHEDULER_SHADOW_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_hints.yaml"
SCHEDULER_SHADOW_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow.yaml"

SCHEDULER_STUB_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub_cvt.yaml"
SCHEDULER_HINTS_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_cvt.yaml"
SCHEDULER_FILTER_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter_cvt.yaml"
SCHEDULER_SHADOW_HINTS_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_hints_cvt.yaml"
SCHEDULER_SHADOW_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_cvt.yaml"

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
  q1-cvt-min)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-cvt"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_STUB="$SCHEDULER_STUB_CVT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_CVT"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-cvt)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-cvt"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_STUB="$SCHEDULER_STUB_CVT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_CVT"
    SCHEDULER_FILTER="$SCHEDULER_FILTER_CVT"
    RUN_FILTER=true
    RUN_SHADOW=false
    ;;
  cvt-shadow)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/cvt-shadow"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_HINTS="$SCHEDULER_SHADOW_HINTS_CVT"
    SCHEDULER_FILTER="$SCHEDULER_SHADOW_CVT"
    RUN_FILTER=false
    RUN_SHADOW=true
    ;;
  *)
    echo "Unknown tier: $TIER" >&2
    echo "Use: pilot|q1-min|q1-full|q1-full-filter|shadow|q1-cvt-min|q1-cvt|q1-cvt-filter|cvt-shadow" >&2
    exit 1
    ;;
esac

apply_long_run_llm_defaults() {
  if [[ -z "${LIFEMANIFOLD_LOG_ITERATION_TIMING:-}" ]]; then
    export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
  fi
  if [[ -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
    export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=4
  fi
}

case "$TIER" in
  q1-min|q1-full|shadow|q1-cvt-min|q1-cvt|cvt-shadow)
    apply_long_run_llm_defaults
    ;;
esac

if [[ "$FILTER_ONLY" == true && -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
  export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2
fi

if [[ ! -f "$BASELINE_ARCHIVE" ]]; then
  echo "Missing baseline archive: $BASELINE_ARCHIVE" >&2
  if [[ "$ARCHIVE_TYPE" == "cvt" ]]; then
    echo "Run: ./scripts/run_cvt_baseline.sh" >&2
  else
    echo "Run: uv run python -m worldspace.scripts.run_map_elites_nightly --archive-type grid" >&2
  fi
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
  echo "Training uncertainty calibration (required for filter/shadow arms)..."
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
  echo "=== tier=$TIER archive=$ARCHIVE_TYPE condition=$condition seed=$seed ==="
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
