# B4 Dungeon factorial — benchmark v2 rerun (n=10)

**Tier:** `q1-v4-dungeon-rerun`  
**Protocol:** [`EXPERIMENT_PROTOCOL_Q1_v4.md`](../../EXPERIMENT_PROTOCOL_Q1_v4.md)  
**Status:** seeds **0–9 DONE** (50/50 runs) · 5,000 proposals/arm · frozen schedulers + `checkpoint_benchmark_v2.pkl`  
**Generated:** 2026-07-18

## Quality gates (all PASS)

| Gate | Target | Result |
|------|--------|--------|
| LLM fallback | ≤5% | **PASS** — stub max **0.99%**, hints max **0.07%**, hints_filter max **0.14%** |
| Filter skip | 25–45% | **PASS** — genetic_filter **36.5%** (35.2–38.5); llm_hints_filter **37.1%** (34.4–40.2) |
| Pilot preserved | yes | `q1-v4-dungeon-pilot` untouched |

## Mean levels at fixed 5,000 proposals (n=10)

| Arm | Cov % | Mean fit | QD-score | Real evals | Skip % | Wall min | Fallback % |
|-----|------:|---------:|---------:|-----------:|-------:|---------:|-----------:|
| **`genetic`** | **54.82 ± 3.11** | 0.7375 | **363.7** | 5,000 | 0.0 | 0.4 | — |
| `genetic_filter` | 50.39 ± 3.13 | 0.7385 | 334.7 | 3,177 | **36.5** | 0.3 | — |
| `llm_stub` | 33.29 ± 1.97 | 0.7335 | 219.5 | 5,000 | 0.0 | 56.9 | 0.14 ± 0.30 |
| `llm_hints` | 35.60 ± 2.21 | 0.7345 | 235.4 | 5,000 | 0.0 | 51.2 | 0.01 ± 0.02 |
| `llm_hints_filter` | 30.22 ± 2.25 | 0.7353 | 199.8 | 3,145 | **37.1** | 54.1 | 0.03 ± 0.05 |

**Headline:** local genetic MAP-Elites **dominates** all LLM arms on coverage/QD (~**+19 pp** vs hints, 10/10 seeds). Mean elite fitness is flat across arms (~0.73–0.74); differences are archive **coverage**, not per-elite quality.

Seed 0 was high for genetic (**62%**); the n=10 mean (**54.8%**) is more representative.

## Matched real-evaluation checkpoint (2,991 evals)

Minimum completed evaluations across all 50 runs. Filter arms skip ~37%, so this is where acquisition effects appear.

| Arm | Cov % | Δ vs genetic |
|-----|------:|-------------:|
| `genetic` | 41.23 ± 1.89 | — |
| **`genetic_filter`** | **49.07 ± 3.02** | **+7.83 pp** (10/10) |
| `llm_stub` | 28.44 ± 1.65 | — |
| `llm_hints` | 29.96 ± 1.85 | +1.52 pp vs stub (8/10) |
| `llm_hints_filter` | 29.87 ± 1.99 | −5.38 pp vs hints at fixed proposals |

At equal simulator budget, **genetic_filter beats genetic** on coverage AUC (Holm reject) — the same sample-efficiency pattern as LifeManifold §3.8. At fixed proposal budget, filter finishes lower (−4.4 pp genetic; −5.4 pp LLM).

## Paired contrasts (fixed proposals, descriptive)

| Contrast | Δcov mean ± SD | Wins | Notes |
|----------|----------------|------|-------|
| hints − stub | **+2.31 ± 3.16 pp** | 9/10 | Modest hint lift; Δfit ≈ 0 |
| genetic_filter − genetic | **−4.43 ± 4.23 pp** | 1/10 | Terminal cost of skipping |
| llm_hints_filter − llm_hints | **−5.38 ± 3.03 pp** | 0/10 | Filter hurts LLM arm here |
| genetic − llm_hints | **+19.22 ± 3.97 pp** | 10/10 | No LLM arm near genetic ME |

## Confirmatory family F-B4-dungeon (Holm m=8, n=10)

Endpoint: coverage / QD-score **AUC** at common budget 2,991 real evaluations.  
Payload: [`v4_dungeon_statistics.json`](v4_dungeon_statistics.json)

| Test | Raw p | Holm @0.05 | Mean ΔAUC | Sign |
|------|-------|------------|-----------|------|
| genetic_filter − genetic (cov AUC) | 0.0010 | **True** | +0.048 | 10/10 |
| genetic_filter − genetic (QD AUC) | 0.0010 | **True** | +29.9 | 10/10 |
| hints − stub (cov AUC) | 0.032 | False | +0.008 | 8/10 |
| hints − stub (QD AUC) | 0.019 | False | +5.3 | 8/10 |
| llm_hints_filter − llm_hints (cov AUC) | 0.50 | False | ≈0 | 6/10 |
| acquisition interaction (cov AUC) | 1.0 | False | −0.048 | 0/10 |

**Verdict: family PASS = False** — only the **no-LLM acquisition filter** contrast survives Holm. Hint content (hints vs stub) shows a consistent but small AUC edge that does not survive correction. LLM+filter interaction is **negative** (filter helps genetic, not LLM).

## Interpretation vs LifeManifold

| Pattern | LifeManifold CA | Dungeon |
|---------|-----------------|---------|
| Bundled LLM+surrogate vs floor | Large (+16–18 pp) | **Fails** — genetic ME wins by ~19 pp |
| hints vs stub (matched acquisition) | ≈ NULL | Small descriptive lift (+2 pp) |
| genetic_filter sample efficiency | PASS @ matched eval | **Replicates** (+7.8 pp @ 2,991 eval; Holm PASS) |
| LLM wall time | ~90 min | ~51–57 min (shorter prompts/domain) |

The dungeon domain is **harder for LLM emitters** (patch JSON, solvability repair) and **easier for archive crossover** (compact 16×16 grid). Engineering stack (retry, filter calibration, fallback logging) **transfers**; the LifeManifold **bundled QD advantage does not**.

## Evaluations to coverage thresholds (descriptive)

Not preregistered before first pilot (protocol v4 §6). Median evals to threshold across seeds:

| Arm | 25% cov | 40% cov | 50% cov |
|-----|--------:|--------:|--------:|
| `genetic` | 775 | 2,025 | 2,975 |
| `genetic_filter` | 625 | 1,400 | 2,400 |
| `llm_stub` | 1,650 | not reached | not reached |
| `llm_hints` | 1,450 | not reached | not reached |
| `llm_hints_filter` | 1,550 | not reached | not reached |

## Verdict

1. **Quality gates:** all PASS on n=10 — safe to report as feasibility + first confirmatory dungeon family.
2. **Acquisition without LLM:** confirmatory **PASS** (matched-eval AUC); replicates LifeManifold filter story on a second domain.
3. **LLM channel:** hints ≳ stub descriptively; neither approaches genetic ME. Not evidence for cross-domain LLM QD lift.
4. **LLM + filter:** terminal coverage **drops**; interaction negative — do not bundle filter under LLM on this domain without redesign.

**Next:** update manuscript §6.13 / protocol §6 gate outcome; optional full 32.5k-proposal budget if cost gate allows.

## Artifacts

- `summary.csv` (50 rows)
- `v4_dungeon_statistics.json`
- `anytime_coverage.png`, `anytime_qd_score.png` (median + IQR, n=10)
- Per-run traces/archives under `{arm}/seed_{0..9}/`
