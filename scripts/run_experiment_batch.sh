#!/usr/bin/env bash
# Run one condition × seed for the Q1 experiment matrix.
# Usage: ./scripts/run_experiment_batch.sh TIER [first_seed] [last_seed]
#
# Grid tiers:  pilot | q1-min | q1-full | q1-full-filter | q1-repeat | shadow
# CVT tiers:   q1-cvt-min | q1-cvt | q1-cvt-filter | cvt-shadow | q1-prompt-ablation
# B2 tier:     q1-v3-pyribs  (CMA-ME + CMA-MAE via run_pyribs_baseline.py)
# v3 G1:       q1-v3-llm-deepseek-v4-pro  (stub+hints; --llm-provider deepseek)
#              q1-v3-llm-gpt-4o-mini       (stub+hints; --llm-provider openai)
#
# q1-repeat: stub+hints only; 3 replicates per seed (default seeds 0–1) for LLM variance floor.
# q1-prompt-ablation: CVT archive + grid system prompt; stub+hints (default seed 0).
# q1-*-filter: filter arm only; requires completed stub + hints for each seed.
# q1-v3-pyribs: seeds × {cma_me,cma_mae}; default 32500 evals; override with PYRIBS_EVALUATIONS (must ÷ 250).
# q1-v3-llm-deepseek-v4-pro / q1-v3-llm-gpt-4o-mini: G1; default seeds 0–4; full: 0 9
#   Requires: DEEPSEEK_API_KEY / OPENAI_API_KEY respectively.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REQUESTED_TIER="${1:-pilot}"
TIER="$REQUESTED_TIER"
SEED_START="${2:-0}"
SEED_END="${3:-$SEED_START}"
FILTER_ONLY=false
RUN_PYRIBS=false
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
PYRIBS_SCRIPT="$ROOT/scripts/run_pyribs_baseline.py"
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
SCHEDULER_STUB_CVT_GRID_PROMPT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub_cvt_grid_prompt.yaml"
SCHEDULER_HINTS_CVT_GRID_PROMPT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_cvt_grid_prompt.yaml"

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
  q1-repeat)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-repeat"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    REPLICATE_COUNT=3
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
  q1-prompt-ablation)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-prompt-ablation"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_STUB="$SCHEDULER_STUB_CVT_GRID_PROMPT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_CVT_GRID_PROMPT"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-v3-pyribs)
    EXP_DIR="$EXP_ROOT/q1-v3-pyribs"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_PYRIBS=true
    ;;
  q1-v3-llm-deepseek-v4-pro)
    # G1: DeepSeek V4 Pro @ official API (non-thinking). stub+hints only.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm/deepseek-v4-pro"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    LLM_PROVIDER=deepseek
    ;;
  q1-v3-llm-gpt-4o-mini)
    # G1: OpenAI gpt-4o-mini (budget). stub+hints only.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm/gpt-4o-mini"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    LLM_PROVIDER=openai
    ;;
  *)
    echo "Unknown tier: $TIER" >&2
    echo "Use: pilot|q1-min|q1-full|q1-full-filter|q1-repeat|shadow|q1-cvt-min|q1-cvt|q1-cvt-filter|cvt-shadow|q1-prompt-ablation|q1-v3-pyribs|q1-v3-llm-deepseek-v4-pro|q1-v3-llm-gpt-4o-mini" >&2
    exit 1
    ;;
esac

# Default LLM provider → worldspace/specs/llm_world_generator_${LLM_PROVIDER}.yaml
LLM_PROVIDER="${LLM_PROVIDER:-qwen}"

REPLICATE_COUNT="${REPLICATE_COUNT:-1}"
if [[ "$REQUESTED_TIER" == "q1-repeat" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=1
fi
if [[ "$REQUESTED_TIER" == "q1-prompt-ablation" && $# -lt 2 ]]; then
  # Prefer documenting seeds 0–2; default remains 0 for cheap resume of existing run.
  SEED_START=0
  SEED_END=0
  echo "NOTE: q1-prompt-ablation default is seed 0 only; for a stronger claim run: $0 q1-prompt-ablation 0 2" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-llm-deepseek-v4-pro" && $# -lt 2 ]]; then
  # G1 minimal default: seeds 0–4 (protocol §6). Full: $0 q1-v3-llm-deepseek-v4-pro 0 9
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-v3-llm-deepseek-v4-pro default seeds 0–4 (G1 minimal); full matrix: $0 q1-v3-llm-deepseek-v4-pro 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-llm-gpt-4o-mini" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-v3-llm-gpt-4o-mini default seeds 0–4 (G1 minimal); full matrix: $0 q1-v3-llm-gpt-4o-mini 0 9" >&2
fi

if [[ "$LLM_PROVIDER" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required for LLM_PROVIDER=deepseek" >&2
  exit 1
fi
if [[ "$LLM_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for LLM_PROVIDER=openai" >&2
  exit 1
fi

apply_long_run_llm_defaults() {
  if [[ -z "${LIFEMANIFOLD_LOG_ITERATION_TIMING:-}" ]]; then
    export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
  fi
  if [[ -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
    export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=4
  fi
}

case "$TIER" in
  q1-min|q1-full|q1-repeat|shadow|q1-cvt-min|q1-cvt|cvt-shadow|q1-prompt-ablation|q1-v3-llm-deepseek-v4-pro|q1-v3-llm-gpt-4o-mini)
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

if [[ "$RUN_PYRIBS" != true ]]; then
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
  local replicate="${4:-}"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -n "$replicate" ]]; then
    out="${out}/rep_${replicate}"
  fi
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
  if [[ -n "$replicate" ]]; then
    extra+=(--replicate "$replicate")
  fi
  echo "=== tier=$TIER archive=$ARCHIVE_TYPE condition=$condition seed=$seed replicate=${replicate:-none} llm=$LLM_PROVIDER ==="
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
      --llm-provider "$LLM_PROVIDER" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_pyribs_one() {
  local algo="$1"
  local seed="$2"
  local out="$EXP_DIR/${algo}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ -n "${PYRIBS_EVALUATIONS:-}" ]]; then
    extra+=(--evaluations "$PYRIBS_EVALUATIONS")
  fi
  echo "=== tier=$TIER archive=$ARCHIVE_TYPE algo=$algo seed=$seed evaluations=${PYRIBS_EVALUATIONS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$PYRIBS_SCRIPT" \
      --algo "$algo" \
      --seed "$seed" \
      --output-dir "$out" \
      --load-archive "$BASELINE_ARCHIVE" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

if [[ "$RUN_PYRIBS" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for algo in cma_me cma_mae; do
      run_pyribs_one "$algo" "$seed"
    done
  done
else
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for rep in $(seq 0 $((REPLICATE_COUNT - 1))); do
      rep_arg=""
      if [[ "$REPLICATE_COUNT" -gt 1 ]]; then
        rep_arg="$rep"
      fi
      if [[ "$RUN_SHADOW" == true ]]; then
        run_one hints "$SCHEDULER_HINTS" "$seed" "$rep_arg"
        run_one filter "$SCHEDULER_FILTER" "$seed" "$rep_arg"
      else
        if [[ "$FILTER_ONLY" == true ]]; then
          require_stub_hints_for_seed "$seed"
        else
          run_one stub "$SCHEDULER_STUB" "$seed" "$rep_arg"
          run_one hints "$SCHEDULER_HINTS" "$seed" "$rep_arg"
        fi
        if [[ "$RUN_FILTER" == true ]]; then
          run_one filter "$SCHEDULER_FILTER" "$seed" "$rep_arg"
        fi
      fi
    done
  done
fi

if [[ -f "$AGG_SCRIPT" ]]; then
  if ! uv run python "$AGG_SCRIPT" --root "$EXP_DIR" --output "$EXP_DIR/summary.csv"; then
    echo "WARNING: failed to write $EXP_DIR/summary.csv (runs may still be valid)" >&2
    exit 1
  fi
  echo "Wrote $EXP_DIR/summary.csv"
fi
