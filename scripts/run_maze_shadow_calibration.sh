#!/usr/bin/env bash
# Calibrate maze filter τ from live QD surrogate archives (25–45% skip band).
#
# Usage:
#   ./scripts/run_maze_shadow_calibration.sh [--dry-run]
#
# Requires completed filter-arm runs with surrogate_archive.jsonl
# (default: q1-v5-maze-pilot genetic_filter + llm_hints_filter seed 0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY=(--apply)
if [[ "${1:-}" == "--dry-run" ]]; then
  APPLY=()
  shift || true
fi

PILOT="$ROOT/artifacts/experiments/q1-v5-maze-pilot"
ARCHIVES=(
  "$PILOT/genetic_filter/seed_0/surrogate_archive.jsonl"
  "$PILOT/llm_hints_filter/seed_0/surrogate_archive.jsonl"
)
for path in "${ARCHIVES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing replay source: $path" >&2
    echo "Run a five-arm pilot first, or pass --surrogate-archive to calibrate_maze_filter.py" >&2
    exit 1
  fi
done

echo "=== Step 1: offline buffer hold-out reference (informational) ==="
uv run python "$ROOT/scripts/calibrate_maze_filter.py" "${APPLY[@]}"

echo "=== Step 2: verify filter skip @ full budget (seed 1, optional) ==="
if [[ ${#APPLY[@]} -gt 0 ]]; then
  uv run python "$ROOT/scripts/run_maze_qd.py" \
    --scheduler "$ROOT/worldspace/specs/maze_scheduler_genetic_filter.yaml" \
    --seed 1 --proposals 5000 \
    --output-dir "$ROOT/artifacts/experiments/q1-v5-maze-calibration-smoke/genetic_filter/seed_1_verify"
fi

echo "Done. Report: artifacts/mazes/surrogate/calibration.json"
