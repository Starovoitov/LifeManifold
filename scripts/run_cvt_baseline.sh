#!/usr/bin/env bash
# Build the CVT warm-start baseline (650 × 50 sims, 0 LLM) for q1-cvt experiments.
# Output: artifacts/map_elites_nightly/cvt/baseline/map_elites_archive.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/artifacts/map_elites_nightly/cvt/baseline"
SCHEDULER="$ROOT/worldspace/specs/map_elites_scheduler_nightly_cvt.yaml"

if [[ -f "$OUT/nightly_run_summary.json" ]]; then
  echo "CVT baseline already complete: $OUT"
  exit 0
fi

echo "=== CVT baseline nightly (no LLM, ~9h sim) ==="
uv run python -m worldspace.scripts.run_map_elites_nightly \
  --archive-type cvt \
  --single-run \
  --scheduler "$SCHEDULER" \
  --output-dir "$OUT"

echo "Done: $OUT/map_elites_archive.jsonl"
