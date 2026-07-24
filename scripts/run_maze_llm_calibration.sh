#!/usr/bin/env bash
# Live maze LLM calibration @ 100 calls (parse/fallback/repair gates).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CALLS="${MAZE_LLM_CALIBRATION_CALLS:-100}"
SEED="${MAZE_LLM_CALIBRATION_SEED:-0}"

echo "=== maze LLM calibration: calls=$CALLS seed=$SEED ==="
uv run python "$ROOT/scripts/calibrate_maze_llm.py" \
  --calls "$CALLS" \
  --seed "$SEED" \
  --output "$ROOT/artifacts/mazes/llm_calibration.json"

echo "Done. Report: artifacts/mazes/llm_calibration.json"
