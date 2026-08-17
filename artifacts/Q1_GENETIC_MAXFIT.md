# Genetic ME `max_fitness_frontier` — descriptive third archive policy

**Status:** complete 2026-08-17 (seeds 0–9, 10/10).
**Tier:** `q1-v3-genetic-me-maxfit` → `artifacts/experiments/q1-v3-genetic-me-maxfit/`
**Scheduler:** `worldspace/specs/map_elites_scheduler_nightly_genetic_me_maxfit.yaml`
**Readout:** [`experiments/q1-v3-genetic-me-maxfit/ANALYSIS.md`](experiments/q1-v3-genetic-me-maxfit/ANALYSIS.md)
**Stack:** 20 random + 30 genetic, surrogate off, LLM off, same 971-niche / 38.84% warm-start JSONL as `genetic_me` / `genetic_me_uniform`, budget 650×50.

Does **not** amend Holm families, locked genetic H2, or the RQ1 LLM 2×2.

## Question

On the genetic ME stack, is the large minfit→uniform coverage movement a
one-sided `min_fitness_frontier` pathology, or does the opposite extreme
(`max_fitness_frontier`) also leave coverage below uniform?

## Result (descriptive)

| Arm | Policy | Mean coverage |
|---|---|---|
| `genetic_me` (frozen) | `min_fitness_frontier` | 45.93 ± 1.50% |
| **`genetic_me_maxfit`** | **`max_fitness_frontier`** | **50.16 ± 1.36%** |
| `genetic_me_uniform` (frozen) | `uniform_frontier` | 59.04 ± 0.76% |

Paired seeds 0–9: maxfit − minfit **+4.23 ± 1.12 pp** (10/10);
maxfit − uniform **−8.87 ± 1.44 pp** (0/10). Descriptive Wilcoxon
two-sided p=0.00195 both contrasts (not Holm).

Visit (maxfit `proposal_log` only; frozen comparators have none):
**4.4 ± 1.6 unique targets** over 32 500 slots (range 2–8);
max-target share 0.57 ± 0.19.

**Reading:** both extremes sit below uniform. The +13.1 pp minfit→uniform
gap is not a one-sided low-fitness pathology. Magnitude is
geometry-specific (\(w_D{=}0.45\), \(\rho(f,D){\approx}0.986\)).
Not a \(w_D\) sweep and not an LLM leftover test.

## Policy

Frontier = occupied cells with at least one empty neighbour.
`max_fitness_frontier` picks the frontier elite with **highest** fitness
(ties → smallest cell id), the mirror of `min_fitness_frontier`.

## Launch (done)

```bash
./scripts/run_experiment_batch.sh q1-v3-genetic-me-maxfit 0 9
```

Log: `artifacts/experiments/q1-v3-genetic-me-maxfit/logs/batch.log`

## Reporting

Descriptive. Not Holm, not TOST, not a new RQ. Primary endpoint: terminal
coverage vs frozen `genetic_me` / `genetic_me_uniform`. Companion: unique
targets / normalized visit entropy from `proposal_log.jsonl` (all emitters).
Manuscript not updated in this readout.
