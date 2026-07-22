# q1-v4-dungeon-full-cpu (Tier 2 supplementary)

**Status:** complete (2026-07-22).  
**Scope:** CPU Role-2 pair only — `genetic` vs `genetic_filter`, seeds 0–9, **32{,}500 proposals**/seed.  
**Confirmatory F-B4** remains on `q1-v4-dungeon-rerun` @ 5k proposals.

## Runs

| Arm | Seeds | Proposals | Mean evals | Mean coverage | Mean skip |
|-----|-------|-----------|------------|---------------|-----------|
| genetic | 10/10 | 32{,}500 | 32{,}500 | 89.42 ± 0.77% | 0% |
| genetic_filter | 10/10 | 32{,}500 | 22{,}480 | 88.79 ± 1.75% | 30.8% |

Wall time: ~59 min total (Tier 2 log 18:13→19:12 MSK).

## Supplementary statistics (`F-B4-dungeon-cpu-full`)

Regenerate:

```bash
.venv/bin/python scripts/analyze_q1_statistics.py \
  --family v4-dungeon-cpu-full \
  --dungeon-root artifacts/experiments/q1-v4-dungeon-full-cpu
```

**AUC @ matched real evaluations (22{,}155 evals; seed 6 filter minimum):**

| Contrast | Mean Δ | Seeds + | Raw p | Holm |
|----------|--------|---------|-------|------|
| genetic_filter − genetic (coverage AUC) | +0.036 | 9/10 | 0.0068 | **Yes** |
| genetic_filter − genetic (QD-score AUC) | +33.9 | 8/10 | 0.0049 | **Yes** |

**Readout:** Role-2 sample-efficiency **ports at full proposal budget** on the evaluation axis, but at **fixed 32.5k proposal slots** filter finishes **−0.63 pp** terminal coverage (mirrors CA Role-2 fixed-iteration trade-off).

## Surrogate skip mechanism (descriptive)

Dungeon filter uses **τ = 0.78** (`dungeon_scheduler_genetic_filter.yaml`), not the CA τ = 0.45.

Aggregated over all `genetic_filter` surrogate logs (10 seeds):

| Budget | Skip rate | Mean pred fitness (skip / eval) | Mean uncertainty (skip / eval) |
|--------|-----------|-----------------------------------|--------------------------------|
| 5k (rerun) | 36.5% | 0.709 / 0.665 | 0.027 / 0.065 |
| 32.5k (full-cpu) | 30.8% | 0.707 / 0.625 | 0.027 / 0.080 |

All skips cite `below_fitness_threshold`; `never_skip_empty_bin` keeps empty-target exploration (≈1.5% of eval rows @ 5k).

## Deferred (journal extension, not arXiv v1)

- LLM arms @ 32.5k (`llm_stub`, `llm_hints`, `llm_hints_filter`)
- `random` baseline @ 32.5k
- One dungeon ablation family (protocol amendment before runs)
