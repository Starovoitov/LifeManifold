# H1 shuffled / permutation placebo (lock)

**Status:** locked for seed-0 pilot  
**Tier:** `q1-h1-placebo-pilot` → `artifacts/experiments/q1-h1-placebo-pilot/`  
**Stack:** LLM mixed emitters (`20R+20G+10L`), `uniform_frontier`, warm-start baseline, checkpoint `nightly_v3_mc_d005`, primary `qwen-turbo`  
**Why:** confirmatory H1 contrasts live parent MLP scalars vs YAML constants `0.5`/`1.0`. That stub is not distribution-matched. This package permutes intact `(fitness, uncertainty)` pairs across the LLM slots in each batch so the scalar multiset matches live hints while parent genotype / few-shot stay with the slot.

## Claim scope (after readout)

Admissible if placebo ≈ live hints (and ≈ stub):

> Under this weak parent-scalar interface, reassigning live MLP scalars across parents does not move coverage relative to matched live hints; a distribution-matched placebo is consistent with the confirmatory near-null.

Admissible if placebo differs from live hints (toward stub or elsewhere):

> Semantic alignment of `(mean, unc)` to the parent in the prompt matters beyond the marginal scalar distribution; the confirmatory stub contrast understates / misattributes the soft channel.

Not claimed either way: general before-generation impossibility; equivalence to hard gating (H2); that the LLM “uses” the numbers without a manipulation check.

## Mechanism

Per iteration, after `prepare_emit` for all LLM slots and **before** HTTP:

1. Collect intact `SurrogatePrediction` objects (live MLP on each slot’s parent).
2. `rng.permutation` reassigns predictions across slots.
3. Rebuild only the surrogate line in each `user_prompt` (`Surrogate predicts fitness ≈ …`); parent JSON and few-shot are untouched.
4. Multiset of `(fitness, uncertainty)` pairs in the batch is preserved.

Requires `performance.llm_parallel_emit: true` (batch prepare). Incompatible with `stub_hints_only` and with `surrogate.enabled: false`.

## Arms

| Condition | Scheduler | Role |
|-----------|-----------|------|
| `hints_placebo` | `map_elites_scheduler_nightly_llm_hints_placebo.yaml` | Shuffled live MLP scalars |
| `hints` (frozen) | `q1-v3-mixed-2x2/hints` or `q1-full/hints` | Live matched baseline (offline compare) |
| `stub_uniform` (frozen) | `q1-v3-mixed-2x2/stub_uniform` | Constant-stub control (offline compare) |

Pilot launches **only** `hints_placebo`. Do not re-run confirmatory hints/stub unless pin mismatch is found.

Held fixed vs confirmatory H1: emitters, warm-start, seeds, checkpoint, budget 650×50, default user prompt template, no after-generation skips.

**Not** in this tier: child rewrite, cold-start, rich/parent/direction prompts.

## Config (locked)

```yaml
llm:
  enabled: true
  hint_placebo: shuffle_batch   # off | shuffle_batch

surrogate:
  enabled: true
  model_type: mlp
  checkpoint: artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl
  # … same calibration / extinction gate as nightly hints
```

Workers for this pilot: `LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2` (thermal).

## Pilot protocol (stop/extend)

| Stage | Seeds | Gate |
|-------|------:|------|
| Pilot | **0 only** | Descriptive readout vs frozen seed-0 `hints` and `stub_uniform` |
| Extend | 0–2 | Only if seed-0 \|Δcov\| (`hints_placebo`−`hints`) ≥ **2 pp** terminal **or** @20k eval (same soft-pilot bar as prior H1 resource screens) |
| Full | 0–9 | Only if extend gate passes; do **not** invent confirmatory TOST mid-flight |

Companion: also note Δfit and QD; do not extend on fitness alone if coverage is flat.

If seed-0 is within ~2 pp of frozen hints → report as negative pilot (distribution-matched placebo consistent with live); do not auto-launch n=10.

## Endpoints (descriptive)

- Terminal coverage / QD / best fitness
- cov@5k / @10k / @20k and AUC cov@20k if `archive_trace` present
- Side-by-side vs frozen seed-0 hints and stub_uniform
- LLM transport: emit attempts, fallbacks (expect ~6500 / 0 like confirmatory)

## Cost

- One arm × seed 0 ≈ 6_500 LLM calls (same as hints)
- Wall: prefer 2 HTTP workers

## Launch

```bash
# Lock + code must be committed first.
LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2 ./scripts/run_experiment_batch.sh q1-h1-placebo-pilot 0 0
```

Requires `QWEN_API_KEY` (source `.env`).

## Preflight

```bash
uv run python -c "
from worldspace.illuminators.scheduler import load_scheduler
c = load_scheduler('worldspace/specs/map_elites_scheduler_nightly_llm_hints_placebo.yaml')
assert c.llm_hint_placebo == 'shuffle_batch'
print('ok', c.llm_hint_placebo, c.surrogate_checkpoint)
"
uv run python -m unittest tests.test_h1_hint_placebo -q
```

## Analysis artifacts (expected)

```
artifacts/experiments/q1-h1-placebo-pilot/
  hints_placebo/seed_0/
  summary.csv          # after aggregate
  ANALYSIS.md          # after human/script readout
```

Compare offline to `artifacts/experiments/q1-v3-mixed-2x2/{hints,stub_uniform}/seed_0/`.

## Manuscript

Wire after readout into `\S\ref{sec:h1-impl-threat}` / Future work. Until pilot completes: locked, not claimed.
