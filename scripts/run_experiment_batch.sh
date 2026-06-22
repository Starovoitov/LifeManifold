#!/usr/bin/env bash
# Run one condition × seed for the Q1 experiment matrix.
# Usage: ./scripts/run_experiment_batch.sh pilot|q1-min|q1-full [first_seed] [last_seed]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TIER="${1:-pilot}"
SEED_START="${2:-0}"
SEED_END="${3:-$SEED_START}"

EXP_ROOT="$ROOT/artifacts/experiments"
BASELINE_ARCHIVE="$ROOT/artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
TRAIN_SCRIPT="$ROOT/scripts/train_surrogate.py"
RUN_SCRIPT="$ROOT/scripts/run_github_llm_map_elites.py"
AGG_SCRIPT="$ROOT/scripts/aggregate_experiment_runs.py"

SCHEDULER_STUB_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub.yaml"
SCHEDULER_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm.yaml"
SCHEDULER_FILTER_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter.yaml"
SCHEDULER_STUB_PILOT="$ROOT/worldspace/specs/map_elites_scheduler_github_llm_stub.yaml"
SCHEDULER_HINTS_PILOT="$ROOT/worldspace/specs/map_elites_scheduler_github_llm.yaml"

case "$TIER" in
  pilot)
    ITERATIONS=120
    EXP_DIR="$EXP_ROOT/pilot"
    SCHEDULER_STUB="$SCHEDULER_STUB_PILOT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_PILOT"
    RUN_FILTER=false
    ;;
  q1-min)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-min"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    ;;
  q1-full)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-full"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    SCHEDULER_FILTER="$SCHEDULER_FILTER_NIGHTLY"
    RUN_FILTER=true
    ;;
  *)
    echo "Unknown tier: $TIER (use pilot|q1-min|q1-full)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$BASELINE_ARCHIVE" ]]; then
  echo "Missing baseline archive: $BASELINE_ARCHIVE" >&2
  echo "Run: uv run python -m worldspace.scripts.run_map_elites_nightly" >&2
  exit 1
fi

CHECKPOINT="$ROOT/artifacts/surrogate/checkpoints/nightly_v2.pkl"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training surrogate checkpoint..."
  uv run python "$TRAIN_SCRIPT" \
    --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
    --checkpoint-path "$CHECKPOINT" \
    --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v2.summary.json" \
    --no-quality-gate
fi

CALIBRATION="$ROOT/artifacts/surrogate/checkpoints/calibration.pkl"
if [[ "$RUN_FILTER" == true && ! -f "$CALIBRATION" ]]; then
  echo "Training uncertainty calibration (required for filter arm)..."
  uv run python "$TRAIN_SCRIPT" \
    --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
    --checkpoint-path "$CHECKPOINT" \
    --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v2.summary.json" \
    --calibrate \
    --calibration-path "$CALIBRATION" \
    --no-quality-gate
fi

mkdir -p "$EXP_DIR"

run_one() {
  local condition="$1"
  local scheduler="$2"
  local seed="$3"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  mkdir -p "$out"
  local extra=()
  if [[ "$condition" == "hints" || "$condition" == "filter" ]]; then
    extra+=(--require-surrogate-quality-gate)
  fi
  echo "=== tier=$TIER condition=$condition seed=$seed ==="
  uv run python "$RUN_SCRIPT" \
    --scheduler "$scheduler" \
    --output-dir "$out" \
    --seed "$seed" \
    --iterations "$ITERATIONS" \
    --load-archive "$BASELINE_ARCHIVE" \
    --llm-provider qwen \
    "${extra[@]}"
}

for seed in $(seq "$SEED_START" "$SEED_END"); do
  run_one stub "$SCHEDULER_STUB" "$seed"
  run_one hints "$SCHEDULER_HINTS" "$seed"
  if [[ "$RUN_FILTER" == true ]]; then
    run_one filter "$SCHEDULER_FILTER" "$seed"
  fi
done

if [[ -f "$AGG_SCRIPT" ]]; then
  uv run python "$AGG_SCRIPT" --root "$EXP_DIR" --output "$EXP_DIR/summary.csv"
  echo "Wrote $EXP_DIR/summary.csv"
fi
