#!/usr/bin/env bash
# Offline + live shadow calibration for the Q1 filter arm (EXPERIMENT_PROTOCOL_Q1.md §4).
#
# Usage:
#   export QWEN_API_KEY=...
#   ./scripts/run_shadow_calibration.sh [seed]
#
# Steps:
#   1. Offline acquisition replay on buffer (skip rate, false_skip estimate)
#   2. Live hints vs shadow run for one seed (archive parity check)
#   3. compare_acquisition_runs hints vs shadow/filter output dirs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${1:-0}"
EXP_DIR="$ROOT/artifacts/experiments/shadow"
SUMMARY="$ROOT/artifacts/surrogate/checkpoints/nightly_v2.summary.json"

echo "=== Step 1: offline acquisition replay (buffer hold-out, yaml thresholds) ==="
uv run python "$ROOT/scripts/report_surrogate_acquisition.py" \
  --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
  --checkpoint-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v2.pkl" \
  --calibration-path "$ROOT/artifacts/surrogate/checkpoints/calibration.pkl" \
  --summary-path "$SUMMARY" \
  --min-predicted-fitness 0.10 \
  --max-uncertainty-to-skip 0.022

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/surrogate/checkpoints/nightly_v2.summary.json").read_text())
acq = summary.get("acquisition") or {}
if not acq:
    print("WARNING: no acquisition block in summary JSON", flush=True)
else:
    print(
        "Offline replay: "
        f"recommended_skip_rate={acq.get('recommended_skip_rate', 0):.1%}, "
        f"false_skip_rate_estimate={acq.get('false_skip_rate_estimate', 0):.1%}, "
        f"calibration_ece={acq.get('calibration_ece', 0):.4f}",
        flush=True,
    )
    skip = float(acq.get("recommended_skip_rate", 0))
    false_skip = float(acq.get("false_skip_rate_estimate", 0))
    print(
        "NOTE: offline replay skip rate is an upper bound; "
        "live shadow target is 25-45% (never_skip_empty_bin + baseline).",
        flush=True,
    )
    if false_skip >= 0.05:
        print(
            f"WARNING: false_skip_rate_estimate {false_skip:.1%} >= 5%",
            flush=True,
        )
PY
fi

echo "=== Step 2: live shadow run (hints + shadow mode, seed=$SEED) ==="
"$ROOT/scripts/run_experiment_batch.sh" shadow "$SEED" "$SEED"

HINTS_DIR="$EXP_DIR/hints/seed_${SEED}"
SHADOW_DIR="$EXP_DIR/filter/seed_${SEED}"

echo "=== Step 3: compare hints vs shadow (archive parity + skip metrics) ==="
uv run python "$ROOT/scripts/compare_acquisition_runs.py" \
  --baseline-dir "$HINTS_DIR" \
  --candidate-dir "$SHADOW_DIR" \
  --grid-resolution 50

echo "Done. Inspect surrogate_archive_shadow.jsonl for per-slot shadow_would_skip counts."
