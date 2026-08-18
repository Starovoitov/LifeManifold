# Maze Phase A — genetic `min_fitness_frontier` vs frozen uniform

**Tier:** `q1-v5-maze-genetic-minfit`
**Protocol:** [`Q1_RQ1_SECOND_DOMAIN.md`](../../../Q1_RQ1_SECOND_DOMAIN.md)
**Status:** seeds **0–9 DONE** (2026-08-17). **GO** for Phase B.
**Comparator:** frozen `q1-v5-maze-full/genetic` (uniform; not re-run).

Does **not** amend Holm H5, CA RQ1, or genetic H2. Not Holm, not TOST.

## Decision (locked cut: 5{,}000 evaluations)

| Arm | Coverage @ 5k % |
|-----|-----------------|
| Frozen `genetic` (uniform) | 56.98 ± 4.47 |
| **`genetic_minfit`** | **16.79 ± 3.29** |
| Δ minfit − uniform | **−40.19 ± 4.13 pp** (10/10) |

Descriptive Wilcoxon two-sided p=0.00195 (stat=0). Protocol GO:
\(\lvert\mathrm{mean}\,\Delta\rvert \ge 5\) pp **and** 10/10 same sign.

Policy moves maze coverage in the headroom regime. A stub/hints 2×2 can
replay the CA inversion *on this evaluator*. Magnitude is not portable
to CA pp.

## Secondary cuts (same paired seeds)

| Evals | Uniform % | Minfit % | Δ pp (10/10) |
|-------|-----------|----------|--------------|
| 8k | 75.51 ± 5.28 | 22.33 ± 4.16 | **−53.18 ± 3.79** |
| 10k | 84.36 ± 5.20 | 24.96 ± 4.41 | **−59.40 ± 3.31** |
| 15k | 97.00 ± 3.05 | 30.37 ± 4.56 | **−66.63 ± 3.26** |
| 32.5k (terminal) | 99.62 ± 1.19 | 45.11 ± 5.39 | **−54.51 ± 4.87** |

Uniform saturates the 900-cell grid; minfit does **not** (terminal range
36.7–52.1%). Terminal is not the claim; it shows the policy gap survives
the H5 ceiling that hid it.

## Per-seed Δ coverage @ 5k (pp)

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| Δ | −42.7 | −35.3 | −46.9 | −40.3 | −39.3 | −34.1 | −41.8 | −42.6 | −43.4 | −35.4 |

## Phase B

Authorized by this GO. Launched 2026-08-17 as `q1-rq1-maze-factorial`
(empty archive, 5k, dated `gpt-4o-mini-2024-07-18`). See
[`Q1_RQ1_SECOND_DOMAIN.md`](../../../Q1_RQ1_SECOND_DOMAIN.md).
