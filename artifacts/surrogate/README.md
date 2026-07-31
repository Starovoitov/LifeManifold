# Surrogate training runbook

Operational commands for buffer hygiene, training, analysis, and checkpoint policy.

Architecture and stage-by-stage design: [`docs/SURROGATE_MODEL.md`](../../docs/SURROGATE_MODEL.md).

## Nightly pipeline order

```text
1. Baseline MAP-Elites (surrogate off)
2. Backfill buffer from baseline archive (append; skip when live_eval + backfill already present)
3. Surrogate-enabled run (append live_eval rows)
4. Train on full buffer -> nightly_v3_mc_d005.pkl + summary + optional calibration
```

```bash
uv run python -m worldspace.scripts.run_map_elites_nightly
```

Training uses `surrogate.model_type` from `worldspace/specs/map_elites_scheduler_nightly_surrogate.yaml` (default `mlp`).

## Buffer filter (dedupe / live-only)

```bash
uv run python scripts/filter_surrogate_buffer.py \
  --input artifacts/surrogate/buffer_nightly.jsonl \
  --output artifacts/surrogate/buffer_live.jsonl \
  --live-only --dedupe
```

Flags:

- `--dedupe` — keep first row per canonical `world_spec` hash
- `--live-only` — `metadata.source == live_eval`
- `--drop-backfill` — remove `archive_backfill` / `archive_backfill_collapsed`
- `--stats-only` — print JSON stats without writing output

## Train

```bash
uv run python scripts/train_surrogate.py \
  --buffer-path artifacts/surrogate/buffer_nightly.jsonl \
  --checkpoint-path artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl \
  --summary-path artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json \
  --consistency-weight 0 \
  --no-quality-gate
```

LightGBM train (legacy):

```bash
uv run python scripts/train_surrogate.py --model-type lightgbm --micro ...
```

Train-time options:

- `--emitter-onehot` — append emitter regime one-hot (27-dim input; not runtime-compatible)
- `--stratify-emitter` — stratified hold-out by `emitter_type`
- `--low-stability-weight 2.0` — up-weight rows with stability < 0.3 (LightGBM only)

## Analyze (parity with train defaults)

```bash
uv run python scripts/analyze_surrogate_buffer.py \
  --buffer artifacts/surrogate/buffer_nightly.jsonl \
  --fit-model --ensemble-size 8 --random-state 42 --test-fraction 0.2 \
  --consistency-weight 0 \
  --output-json artifacts/surrogate/buffer_analysis.json
```

Compare LightGBM vs MLP: `--compare-models --fit-model`.

## Tiered quality gate

Summary fields:

| Field | Threshold | Use |
|-------|-----------|-----|
| `hints_ok` | R²(fitness) ≥ 0.30, MAE(fitness) < 0.085 | LLM pilot hints (`checkpoint_quality_allows_hints`) |
| `quality_passed` | R²(fitness) > 0.72 + MAE gates | production policy |

Nightly train uses `--no-quality-gate` so checkpoints are written for pilot-tier models. Runtime stub fallback still applies when `require_quality_gate` is enabled and `hints_ok` is false.

## Schema 2.1 migrate

```bash
uv run python scripts/migrate_surrogate_buffer.py \
  --buffer artifacts/surrogate/buffer.jsonl \
  --re-featurize --target-schema 2.1 \
  --output artifacts/surrogate/buffer_v21.jsonl
```

## Eval memo (schema v3)

```bash
uv run python scripts/record_surrogate_improvement_eval.py \
  --summary artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json \
  --buffer-path artifacts/surrogate/buffer_nightly.jsonl \
  --dataset-source nightly_append \
  --notes "cw=0, full buffer"
```

Template: `artifacts/surrogate/surrogate_improvement_eval.template.json`.

## Experimental MLP scheduler

CI / smoke spec: `worldspace/specs/map_elites_scheduler_mini_surrogate_mlp.yaml` (`surrogate.model_type: mlp`).
