# Sphere RQ1 Phase B — LLM 2×2 (empty archive, 5k)

**Tier:** `q1-rq1-sphere-factorial`
**Protocol:** [`Q1_RQ1_SPHERE_DOMAIN.md`](../../../Q1_RQ1_SPHERE_DOMAIN.md)
**Status:** seeds **0–9 DONE** (2026-08-26 23:51 +03). **40/40.**
**Model:** dated `gpt-4o-mini-2024-07-18`; call logs on all 40 runs.
**Workers:** 2; launched 2026-08-26 15:43 +03 after live preflight 20/20
parse 1.0 / fallback 0 / mean L2 0.68.

Does **not** amend Holm, CA H1, maze empty 2×2, or Sphere H2.
Not Holm, not TOST. Does not identify CA Holm $+15.09$ or maze $+39$ pp.
Empty $100\times100$ is not occupancy-matched to the CA $971$-niche /
$38.84\%$ floor. **Journal PDF: appendix `tab:sphere-h1` only**
(2026-08-27 Reporting). Not main text.

## Primary cut (terminal coverage @ 5{,}000 proposals)

| Cell | Prompt | Policy | Coverage % |
|------|--------|--------|------------|
| `llm_stub_minfit` | stub | minfit | **10.26 ± 0.12** |
| `llm_stub_uniform` | stub | uniform | **12.94 ± 0.23** |
| `llm_hints_minfit` | live | minfit | **10.26 ± 0.15** |
| `llm_hints_uniform` | live | uniform | **12.73 ± 0.23** |

± is population SD across seeds (same as `analyze_rq1_sphere_factorial.py`).
All 40: 5000 evaluations, dated model ID. Parse $\ge 99.99\%$; fallback
$\le 0.01\%$. Mean delta L2 $\approx 0.58$–$0.71$. Leftover is **not**
genetic-Gaussian contamination.

## Paired contrasts (n=10, seeds 0–9)

| Contrast | Mean Δ ± SD | Sign $+$/−/$0$ | Paired $t$ 95% CI |
|----------|-------------|----------------|-------------------|
| policy @ stub (`uniform` − `minfit`) | **+2.68 ± 0.28 pp** | **10/0/0** | $[+2.47,+2.89]$ |
| policy @ live | **+2.47 ± 0.28 pp** | **10/0/0** | $[+2.26,+2.68]$ |
| leftover @ minfit (live − stub) | **+0.00 ± 0.04 pp** | 6/4/0 | $[-0.03,+0.04]$ |
| leftover @ uniform | **−0.21 ± 0.16 pp** | 1/9/0 | $[-0.33,-0.09]$ |
| interaction (leftover@uni − leftover@minfit) | $-0.21 \pm 0.17$ pp | 0/10/0 | $[-0.33,-0.09]$ |

CI uses sample SD and $t_{9,0.975}$. Analyzer ± uses population SD.

Off-diagonal bundle `hints_uniform` − `stub_minfit` $= +2.47$ pp and equals
$\Delta_{\mathrm{policy}\mid\mathrm{stub}}+\Delta_{\mathrm{soft}\mid\mathrm{uniform}}$
on cell means ($2.68-0.21$).

## Per-seed coverage %

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| stub minfit | 10.17 | 10.37 | 10.27 | 10.13 | 10.36 | 10.05 | 10.23 | 10.42 | 10.17 | 10.39 |
| stub uniform | 12.68 | 12.68 | 12.94 | 13.38 | 13.04 | 12.84 | 13.30 | 12.97 | 12.74 | 12.79 |
| hints minfit | 10.19 | 10.41 | 10.25 | 10.09 | 10.39 | 10.04 | 10.25 | 10.45 | 10.07 | 10.45 |
| hints uniform | 12.57 | 12.70 | 12.76 | 13.30 | 12.81 | 12.59 | 12.72 | 12.90 | 12.42 | 12.53 |

## Reading

Same qualitative split as CA and maze: **policy moves coverage, parent-level
scalars do not add a second policy-sized channel.** Matching
`target_selection` leaves leftover $+0.00$ pp at minfit. At uniform the
leftover is a small **negative** (live below stub; 9/10; CI excludes 0).
That is not a hints win and not a new Holm family. It is still an order of
magnitude below the policy $\Delta$ (${\approx}{+}2.5$ pp). Do not read
$-0.21$ pp as portable “scalars hurt.”

Magnitudes are **not** portable. Genetic Phase A @ 5k was
**−5.79 ± 0.55 pp** (minfit $13.59\%$ vs uniform $19.38\%$). The LLM policy
gap is the same sign, compressed (${\approx}2.5$–$2.7$ pp). LLM sits
**below** genetic at both policies at this cut (uniform $12.7$–$12.9$ vs
$19.4$; minfit $10.3$ vs $13.6$). Local-delta actuator, not a ranking of
emitters. Absolute coverage ${\approx}10$–$13\%$ on a 10k-cell archive
leaves headroom; the 5k lock matches the maze empty $2\times2$, not Sphere
H2 (32.5k, no LLM).

Sphere H2 `me_uniform` / `me_filter` remain the wrong cells for this story.

Reproduce: `python scripts/analyze_rq1_sphere_factorial.py`
