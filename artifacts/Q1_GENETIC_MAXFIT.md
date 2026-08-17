# Genetic ME `max_fitness_frontier` — descriptive third archive policy

**Status:** wiring only; **no experiment launched**.
**Tier:** `q1-v3-genetic-me-maxfit` → `artifacts/experiments/q1-v3-genetic-me-maxfit/`
**Scheduler:** `worldspace/specs/map_elites_scheduler_nightly_genetic_me_maxfit.yaml`
**Stack:** 20 random + 30 genetic, surrogate off, LLM off, same 971-niche / 38.84% warm-start JSONL as `genetic_me` / `genetic_me_uniform`, budget 650×50.

Does **not** amend Holm families, locked genetic H2, or the RQ1 LLM 2×2.

## Question

On the genetic ME stack, is the large minfit→uniform coverage movement a
one-sided `min_fitness_frontier` pathology, or does the opposite extreme
(`max_fitness_frontier`) also leave coverage below uniform?

Frozen comparators (do not re-run):

| Arm | Policy | Mean coverage |
|---|---|---|
| `genetic_me` | `min_fitness_frontier` | 45.93% |
| `genetic_me_uniform` | `uniform_frontier` | 59.04% |

This arm adds `genetic_me_maxfit`. Under \(w_D{=}0.45\) / \(\rho(f,D){\approx}0.986\),
maxfit is expected to concentrate on high-\(D\) frontier cells (visit entropy
near minfit, not uniform). Magnitude of \(\Delta\) vs uniform is geometry-specific;
this is not a \(w_D\) sweep and not an LLM leftover test.

## Policy

Frontier = occupied cells with at least one empty neighbour.
`max_fitness_frontier` picks the frontier elite with **highest** fitness
(ties → smallest cell id), the mirror of `min_fitness_frontier`.

## Launch (not executed)

```bash
# n=10, CPU only; ~6 min/seed, ~1 h serial. All-emitter proposal_log is on.
./scripts/run_experiment_batch.sh q1-v3-genetic-me-maxfit 0 9
```

Smoke:

```bash
./scripts/run_experiment_batch.sh q1-v3-genetic-me-maxfit 0 0
```

## Reporting

Descriptive. Not Holm, not TOST, not a new RQ. Primary endpoint: terminal
coverage vs frozen `genetic_me` / `genetic_me_uniform`. Companion: unique
targets / normalized visit entropy from `proposal_log.jsonl` (all emitters).
