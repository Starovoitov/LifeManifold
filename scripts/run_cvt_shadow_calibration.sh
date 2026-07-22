#!/usr/bin/env bash
# Offline + live CVT shadow calibration before q1-cvt filter arm.
# Usage: ./scripts/run_cvt_shadow_calibration.sh [seed]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${1:-0}"
EXP_DIR="$ROOT/artifacts/experiments/cvt-shadow"
CVT_BASELINE="$ROOT/artifacts/map_elites_nightly/cvt/baseline/map_elites_archive.jsonl"

if [[ ! -f "$CVT_BASELINE" ]]; then
  echo "Missing CVT baseline: $CVT_BASELINE" >&2
  echo "Run: ./scripts/run_cvt_baseline.sh" >&2
  exit 1
fi

echo "=== Step 1: offline acquisition replay (same thresholds as grid filter) ==="
uv run python "$ROOT/scripts/report_surrogate_acquisition.py" \
  --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
  --checkpoint-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl" \
  --calibration-path "$ROOT/artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl" \
  --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json" \
  --min-predicted-fitness 0.45 \
  --max-uncertainty-to-skip 1.0

echo "=== Step 2: live CVT shadow run (hints + shadow mode, seed=$SEED) ==="
HINTS_DIR="$EXP_DIR/hints/seed_${SEED}"
SHADOW_DIR="$EXP_DIR/filter/seed_${SEED}"

if ! "$ROOT/scripts/run_experiment_batch.sh" cvt-shadow "$SEED" "$SEED"; then
  batch_rc=$?
  for run_dir in "$HINTS_DIR" "$SHADOW_DIR"; do
    if [[ ! -f "$run_dir/nightly_run_summary.json" ]]; then
      echo "ERROR: CVT shadow run incomplete: $run_dir" >&2
      exit "$batch_rc"
    fi
  done
  echo "CVT shadow runs complete; continuing despite batch post-processing failure." >&2
fi

echo "=== Step 3: compare hints vs shadow (archive parity + skip metrics) ==="
uv run python "$ROOT/scripts/compare_acquisition_runs.py" \
  --baseline-dir "$HINTS_DIR" \
  --candidate-dir "$SHADOW_DIR" \
  --n-cells 2500

AGG_SCRIPT="$ROOT/scripts/aggregate_experiment_runs.py"
if [[ -f "$AGG_SCRIPT" ]]; then
  uv run python "$AGG_SCRIPT" --root "$EXP_DIR" --output "$EXP_DIR/summary.csv"
  echo "Wrote $EXP_DIR/summary.csv"
fi

echo "Done. If live skip rate diverges from grid (25-45%), retune thresholds before q1-cvt filter."
