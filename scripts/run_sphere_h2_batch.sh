#!/usr/bin/env bash
# Supplementary Sphere H2: me_uniform vs me_filter, seeds 0-9.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/experiments/q1-v3-sphere-h2"
CKPT="$ROOT/artifacts/surrogate/sphere_h2_mlp.joblib"
PY="$ROOT/.venv/bin/python"
mkdir -p "$OUT"/{me_uniform,me_filter}

"$PY" "$ROOT/scripts/run_sphere_h2.py" train --target-skip 0.40 --out "$CKPT"

for seed in $(seq 0 9); do
  for arm in me_uniform me_filter; do
    dest="$OUT/$arm/seed_$(printf '%02d' "$seed")"
    mkdir -p "$dest"
    echo "=== $arm seed=$seed ==="
    "$PY" "$ROOT/scripts/run_sphere_h2.py" run \
      --arm "$arm" \
      --seed "$seed" \
      --proposals 32500 \
      --surrogate "$CKPT" \
      --output-dir "$dest"
  done
done

"$PY" "$ROOT/scripts/analyze_sphere_h2.py" --root "$OUT"
