# LifeManifold Q1 v4 — B4 Dungeon factorial protocol

**Created / frozen:** 2026-07-17  
**Status:** seeds **0–9 DONE** (50/50 runs @ 5k proposals); quality gates PASS; F-B4-dungeon **PARTIAL** (2026-07-18)  
**Relationship to v3:** new cross-domain family; does not amend any v2/v3 Holm family.

## 1. Objective

Test which parts of the LLM–surrogate stack transfer to a second structured
QD domain. The domain is a self-contained dungeon generator, not the
LifeManifold CA and not the B3 analytic vector benchmark.

## 2. Locked domain

| Knob | Value |
|------|-------|
| Candidate | 16×16 JSON tile rows |
| Tiles | `# . S G K D H` |
| Constraints | wall boundary; exactly one start/goal; zero or one key-door pair |
| Planner | stateful A* over `(position, has_key)` |
| Hazards | 16 candidate-hash-seeded rollouts; each hazard blocks with p=0.35 |
| Fitness | `0.55 robustness + 0.30 BC1 + 0.15 reachable_ratio`, clipped [0,1]; unsolvable = 0 |
| BC1 | `(shortest_path − 10) / 40`, clipped [0,1] |
| BC2 | `(reachable_junction_density − 0.25) / 0.50`, clipped [0,1] |
| Archive | 30×30 grid; one highest-fitness elite per cell |
| Parent selection | `uniform_frontier` for every arm |
| Initialization | 100 shared random proposals |
| Full budget | 32,500 proposals per arm/seed |
| Seeds | 0–9; 0–4 exploratory gate, 5–9 extension |

## 3. Locked arms

| Arm | Emitters after warm-up | LLM | Surrogate prompt | Acquisition |
|-----|------------------------|-----|------------------|-------------|
| `genetic` | 20 random + 30 genetic | off | off | off |
| `genetic_filter` | 20 random + 30 genetic | off | prediction | threshold gate |
| `llm_stub` | 20 random + 30 LLM | on | constants | off |
| `llm_hints` | 20 random + 30 LLM | on | prediction | off |
| `llm_hints_filter` | 20 random + 30 LLM | on | prediction | threshold gate |

All arms share proposal slots, target selection, archive geometry, initial
generator, and seed indices. Filter arms are reported at both fixed proposal
and fixed real-evaluation budgets.

## 4. Surrogate lock and gates

- Training rows come from reserved random/genetic design seeds and cannot use
  experiment seeds.
- Feature schema is frozen and target-free; direct heads predict fitness,
  BC1, and BC2.
- Quality gate: hold-out fitness MAE must beat the training-mean predictor and
  Spearman correlation must be positive.
- Acquisition threshold is selected by shadow replay for 25–45% skip,
  `never_skip_empty_bin=true`.
- Live/replay action divergence must be ≤0.05 before filter confirmation.

**Frozen surrogate gate (2026-07-17):** 2,000 reserved rows; hold-out
fitness MAE 0.0437 vs mean baseline 0.0653; Spearman 0.632. Live-shadow
threshold 0.75 with uncertainty cap 0.03355 yields 31.4% recommended skip;
replay divergence 0.000. All gates pass.

## 5. LLM gates

- Strict JSON parser and schema validation; invalid generations use a logged
  deterministic genetic fallback.
- Preflight parse success ≥95%, nonzero parent–child tile distance, and repair
  collapse <10%.
- Model config, prompt hashes, fallback rate, latency, and API-call count are
  recorded in every run summary.

**Preflight (2026-07-17):** patch-based JSON format, qwen provider, 20 calls;
19/20 valid nonzero mutations (95%), 5% fallback, mean 4.84 changed tiles,
prompt hash `4d25cb08b88ea3f4`. All frozen gates pass.

## 6. Execution gates

1. Unit/golden tests.
2. Random/genetic baseline smoke and non-degenerate archive.
3. Surrogate hold-out + shadow acquisition.
4. LLM mocked tests and paid preflight.
5. Five-arm 250-proposal smoke.
6. Five-arm seed-0 5k pilot.
7. Seeds 0–4 exploratory matrix. **DONE 2026-07-18**
8. Seeds 5–9 only if all frozen quality/cost gates pass. **DONE 2026-07-18**

Failed gates stop downstream execution and remain preserved as artifacts.
The cost gate stops expansion when the measured pilot projects either more
than **100,000 paid LLM calls** or more than **24 serial-provider hours** for
the next matrix stage. A stopped matrix is reported as feasibility evidence,
not as a statistical verdict.

### Gate outcome (2026-07-17)

**Original pilot** (`q1-v4-dungeon-pilot`, preserved): genetic 62.00%; LLM-stub
45.78%; LLM-hints 52.00%. Filter skip only 19.8–20.3%; LLM fallback 28–34%.
Expansion initially stopped.

**Rerun seed-0** (`q1-v4-dungeon-rerun`, 2026-07-17 evening): guided retry +
solvable-repair LLM path; separate dungeon calibration
`checkpoint_benchmark_v2.pkl` (fitness 0.78, uncertainty 0.04; LifeManifold CA
surrogate unchanged). Genetic 62.00%; genetic_filter 51.89% @ **35.3% skip**;
LLM-stub 33.11% @ **0% fallback**; LLM-hints 36.67% @ **0.07% fallback**;
LLM-hints-filter 35.00% @ **34.5% skip / 0.14% fallback**. Quality gates PASS.
At matched 3,235 evaluations, genetic_filter exceeds genetic (+4.1 pp). Seed-0
only — superseded by n=10 matrix below.

**Protocol deviation:** numerical coverage thresholds were not written before
the feasibility pilot, so evaluations-to-threshold are not presented as a
preregistered endpoint. Thresholds 0.25, 0.40, and 0.50 were frozen before
seeds 1–9 resumed.

### Gate outcome (2026-07-18) — n=10 matrix complete

**Matrix:** `q1-v4-dungeon-rerun`, seeds **0–9**, 5,000 proposals/arm, 50/50 runs.
Frozen schedulers + `checkpoint_benchmark_v2.pkl` unchanged from seed-0 rerun.

**Quality gates (all PASS):**

| Gate | Target | n=10 result |
|------|--------|---------------|
| LLM fallback | ≤5% | stub max **0.99%**; hints max **0.07%**; hints_filter max **0.14%** |
| Filter skip | 25–45% | genetic_filter **36.5%** (35.2–38.5); llm_hints_filter **37.1%** (34.4–40.2) |

**Mean fixed-proposal levels (n=10):**

| Arm | Cov % | QD-score | Skip % | Fallback % |
|-----|------:|---------:|-------:|-----------:|
| `genetic` | **54.82 ± 3.11** | **363.7** | 0 | — |
| `genetic_filter` | 50.39 ± 3.13 | 334.7 | 36.5 | — |
| `llm_stub` | 33.29 ± 1.97 | 219.5 | 0 | 0.14 |
| `llm_hints` | 35.60 ± 2.21 | 235.4 | 0 | 0.01 |
| `llm_hints_filter` | 30.22 ± 2.25 | 199.8 | 37.1 | 0.03 |

**Matched real-evaluation checkpoint (2,991 evals):** genetic_filter beats genetic
**+7.83 pp** (10/10); hints beats stub **+1.52 pp** (8/10). At fixed proposals,
filter arms finish lower (genetic_filter **−4.43 pp** vs genetic).

**F-B4-dungeon (Holm m=8, AUC @ 2,991 evals):** `genetic_filter − genetic`
coverage and QD-score AUC **Holm reject** (p≈0.001, 10/10). `llm_hints − llm_stub`
AUC positive descriptively (raw p≈0.02–0.03) but **does not survive Holm**.
LLM+filter interaction **negative** (0/10). **Family verdict: PARTIAL FAIL**
(`family_pass=false`; only acquisition-without-LLM confirmatory).

**Interpretation:** engineering stack transfers (fallback, skip band, retry);
LifeManifold **bundled LLM QD lift does not** — genetic ME **+19.2 pp** over hints
(10/10). LLM-vs-genetic contrasts remain **descriptive** per §7.

Analysis: [`experiments/q1-v4-dungeon-rerun/ANALYSIS.md`](experiments/q1-v4-dungeon-rerun/ANALYSIS.md),
[`v4_dungeon_statistics.json`](experiments/q1-v4-dungeon-rerun/v4_dungeon_statistics.json).
Optional extension: full 32.5k-proposal budget if cost gate allows — see **§9 tiers** (recommended: Tier 2 CPU only; Tier 4 not recommended).

## 7. Statistical family

Primary incremental contrasts:

1. `llm_hints − llm_stub` — prompt hint content.
2. `genetic_filter − genetic` — acquisition without LLM.
3. `llm_hints_filter − llm_hints` — acquisition with LLM.
4. Difference-in-differences of the two acquisition effects.

Endpoints are coverage AUC and QD-score AUC at a common real-evaluation
budget equal to the **minimum completed evaluation count** across the five
arms and $n{=}10$ seeds (observed: **2,991**). Per seed, the
coverage/QD-vs-evaluations series is linearly interpolated onto a step-50
grid, integrated with the trapezoid rule, and divided by that horizon
(normalized AUC). Confirmatory tests are paired one-sided Wilcoxon on
**per-seed ΔAUC**, not AUC of an aggregate median curve. Filter arms are
not last-observation-carried-forward past their final logged evaluation
into the Holm integral. Final fixed-proposal coverage, mean fitness,
QD-score, evaluations, wall time, and fallback/skip rates are secondary.
Paired Wilcoxon, paired A₁₂, bootstrap intervals, and Holm correction
apply only within this v4 family. LLM-vs-genetic and all CA/B3
comparisons are descriptive.

**Outcome (2026-07-18, n=10):** common budget **2,991** real evaluations.
Holm family **m=8**. Only **`genetic_filter − genetic`** (coverage AUC and
QD-score AUC) survives correction. **`llm_hints − llm_stub`** does not.
Difference-in-differences interaction **fails** (negative on all seeds).
Payload: `artifacts/experiments/q1-v4-dungeon-rerun/v4_dungeon_statistics.json`.
Run: `uv run python scripts/analyze_q1_statistics.py --family v4-dungeon --dungeon-root artifacts/experiments/q1-v4-dungeon-rerun`.

---

## 8. Reviewer FAQ — generality / budget asymmetry *(disclosure; 2026-07-21)*

**Q: Are results an artifact of one custom CA simulator?**

| Issue | Answer |
|-------|--------|
| **Primary confirmatory domain** | LifeManifold **CA only** for v2/v3 Holm families (F-RQ0/1/4). We do **not** claim domain-agnostic LLM QD from CA alone. |
| **Second domain (B4)** | Self-contained **dungeon** (16×16 tile JSON + A* planner) — not CA, not B3 analytic benchmarks. n=10 factorial, F-B4-dungeon **PARTIAL**. |
| **What transfers** | Engineering gates (fallback ≤5%, skip 25–45%) + **`genetic_filter − genetic` AUC Holm PASS** (Role-2 / acquisition without LLM). |
| **What does not** | Bundled LLM+surrogate QD lift: genetic ME **+19.2 pp** over hints (10/10); hints−stub AUC **does not survive Holm**; LLM+filter interaction **negative**. |
| **CVT / B3** | **CVT** = same CA simulator, different archive geometry (sensitivity only). **B3 sphere/rastrigin** = pyribs runner sanity — **no LLM/surrogate arms** (protocol v3 §3.12). |
| **Paper object (2026-07-19)** | *Two surrogate roles* (before vs after generation) — B4 supports **filter portability**, not cross-domain LLM emitter superiority. |

**Q: Is B4 a “full replication” of the CA matrix?**

| Issue | Answer |
|-------|--------|
| **Preregistered full budget** | **32,500 proposals** / arm / seed (§2). |
| **Executed matrix (frozen)** | **5,000 proposals** / arm / seed — **×6.5 shorter** than preregistered full budget. |
| **Primary B4 endpoints** | Coverage / QD-score **AUC** at **matched real-evaluation budget** (2,991 evals across 50 runs) — not terminal coverage @ 32.5k. |
| **Protocol deviation** | Coverage-threshold race endpoints **not** preregistered before first pilot (§6); disclosed; not used as confirmatory gates. |
| **Reviewer framing** | B4 is **cross-domain component transfer**, not matched-budget replication of CA bundled LLM gains. Budget asymmetry must appear in Limitations + this FAQ. |

**Suggested rebuttal (one paragraph):**

> We do not claim universal LLM QD beyond our simulators. Primary confirmatory tests remain on LifeManifold CA. B4 (dungeon, n=10) tests the same surrogate-integration stack on a second structured domain with featurizable genomes: acquisition without LLM replicates at matched evaluation AUC (Holm), while bundled LLM+surrogate QD lift does not (genetic ME dominates; PARTIAL family). B4 used 5k proposals (not preregistered 32.5k) under the v4 cost gate; we report matched-eval AUC and disclose the budget gap explicitly.

**Manuscript:** Limitations §7.4 (single confirmatory CA domain + B4 partial transfer + budget gap); Results §6.11; appendix Table J (B4 budget note); rebuttal block in outline.

---

## 9. Optional compute extensions — tiers & time estimates *(from n=10 @ 5k; 2026-07-21)*

Observed wall times on `q1-v4-dungeon-rerun` @ **5,000 proposals** (see `summary.csv`):

| Arm class | Mean wall / seed | LLM calls / seed | Notes |
|-----------|------------------:|-----------------:|-------|
| `genetic`, `genetic_filter` | **~0.3–0.4 min** | 0 | CPU-only; A* + surrogate predict |
| `llm_stub`, `llm_hints`, `llm_hints_filter` | **~51–57 min** | **~3,400–3,800** | API-bound |

Linear scaling to **32,500 proposals** (6.5×): genetic ~**2–3 min**/seed; LLM ~**5.5–6 h**/seed (~**23k calls**/seed). v4 cost gate: **≤100k LLM calls** or **≤24 serial-provider hours** per expansion stage.

| Tier | What to run | Runs | Est. wall (serial) | Est. LLM calls | Expected payoff | Recommend |
|------|-------------|-----:|-------------------:|---------------:|-----------------|:---------:|
| **0 — disclosure only** | Protocol FAQ + Limitations text (§8) | 0 | **~1–2 h** author | 0 | Closes wording gap | **Before submit** |
| **1 — re-analyze existing** | Regenerate stats / Fig. B4 from frozen 5k matrix | 0 | **~30 min** | 0 | Tables current | **Done** |
| **2 — CPU full budget (Role-2)** | `genetic` + `genetic_filter`, seeds 0–9, `DUNGEON_PROPOSALS=32500` → `q1-v4-dungeon-full-cpu/` | 20 | **~1 h** | 0 | Role-2 AUC @ full proposal horizon | **DONE 2026-07-22** |
| **3 — LLM hint pilot full budget** | `llm_hints` + `llm_stub`, seeds **0–4 only**, 32.5k | 10 | **~55 h** serial | **~230k** | Unlikely to flip Holm | **Deferred** → §11 journal extension |
| **4 — full 5×10 @ 32.5k** | All arms, seeds 0–9 | 50 | **~165 h** LLM + **~1 h** CPU | **~700k** | Exceeds cost gate | **Not recommended**; LLM subset in §11 |

**Tier 2 outcome (2026-07-22):** 20/20 runs complete; `v4_dungeon_cpu_full_statistics.json`; supplementary Holm $m{=}2$ **PASS** (coverage + QD AUC @ 22{,}155 matched evals); terminal @ fixed 32.5k proposals genetic **89.42 ± 0.77%** vs.\ filter **88.79 ± 1.75%** ($-0.63$~pp). Analysis: [`experiments/q1-v4-dungeon-full-cpu/ANALYSIS.md`](experiments/q1-v4-dungeon-full-cpu/ANALYSIS.md).

**Commands (Tier 2 — done):**

```bash
export DUNGEON_EXPERIMENT_ROOT=q1-v4-dungeon-full-cpu
export DUNGEON_PROPOSALS=32500
./scripts/run_experiment_batch.sh q1-v4-dungeon-genetic 0 9
./scripts/run_experiment_batch.sh q1-v4-dungeon-genetic-filter 0 9
uv run python scripts/analyze_q1_statistics.py --family v4-dungeon-cpu-full \
  --dungeon-root artifacts/experiments/q1-v4-dungeon-full-cpu
```

**Author-time checklist (Tier 0, no new runs):**

- [x] Limitations: B4 budget 5k vs 32.5k + matched-eval AUC primary → outline §7.4 (2026-07-22)
- [x] Results §6.11: PARTIAL family + genetic dominance + budget disclosure (2026-07-22)
- [x] Rebuttal paste from §8 FAQ → `SWARM_EC_DRAFT_OUTLINE.md` rebuttal block (2026-07-22)
- [x] Optional: Tier 2 before final accept if reviewer insists on full-budget evidence → **DONE 2026-07-22** (supplementary CPU; arXiv v1)

**Does not** amend v2/v3 Holm families or redefine F-B4 primary endpoints without §10 amendment.

---

## 11. Journal extension — draft amendment *(NOT FROZEN; execute only after arXiv v1)*

**Draft date:** 2026-07-22  
**Status:** planning only — **do not run** until this section is copied into §10 with a freeze date and `DUNGEON_EXPERIMENT_ROOT` is set.

### 11.1 Purpose

Extend B4 evidence for the **journal revision** (post--arXiv v1). This extension **does not** replace confirmatory F-B4 @ 5k (`q1-v4-dungeon-rerun`). It adds supplementary full-budget runs and one ablation family under separate Holm labels.

### 11.2 Locked scope (package C)

| Block | Arms | Seeds | Proposals | Artifact root | Est. wall |
|-------|------|-------|-----------|---------------|-----------|
| **C1 — LLM full budget** | `llm_stub`, `llm_hints`, `llm_hints_filter` | 0–9 | 32{,}500 | `q1-v4-dungeon-full-llm/` | ~175 h serial LLM |
| **C2 — random baseline** | `random` | 0–9 | 32{,}500 | same tree or `q1-v4-dungeon-full-llm/` | ~30 min CPU |
| **C3 — threshold ablation** | `genetic_filter` @ $\tau{=}0.70$ vs locked $\tau{=}0.78$ | 0–9 each | 32{,}500 | `q1-v4-dungeon-filter-tau-ablation/` | ~1 h CPU |
| **C4 — discrete CMA ceiling (CA grid)** | Bernoulli/categorical CMA-ME vs frozen continuous `cma_me` + matched `hints` / `genetic_me_uniform` | 0–4 gate, then 0–9 | 32{,}500 | `q1-v3-pyribs-discrete-cma/` | ~CPU hours (no LLM) |

**C4 intent:** retire the continuous-relaxation apples-to-oranges caveat on the coverage ceiling. Implement via pyribs custom emitter **or** Gaussian proposal + per-bit Bernoulli decode that stores discrete rule bits in the evaluated genotype (same warm-start JSONL, same 21-D interface scalars). Primary read: does discrete CMA still lead the mid-band, and by how many pp vs continuous CMA-ME? **Not** a new confirmatory family replacing F-RQ-ceiling unless amended; default = supplementary / descriptive paired contrasts, with optional Holm only if §10 freezes a family ID before runs.

**Already complete (arXiv v1 supplementary, not part of §11):** `genetic` + `genetic_filter` @ 32.5k in `q1-v4-dungeon-full-cpu/`; continuous CMA encode ablation `q1-cma-encoding-ablation/` (seeds 0–4).

**Not in scope:** full 5×10 @ 32.5k in one confirmatory family; CMA on dungeon; multi-provider dungeon.

### 11.3 Endpoints (supplementary families)

| Family ID | Contrasts | Holm? | Primary metrics |
|-----------|-----------|-------|-----------------|
| **F-B4-dungeon-llm-full** | Same four contrasts as §7 @ **matched real-eval budget** on 32.5k matrix | Yes, $m{=}8$ | Coverage AUC, QD-score AUC |
| **F-B4-dungeon-random** | `genetic − random` terminal coverage @ 32.5k | Descriptive only | Terminal coverage, QD-score |
| **F-B4-dungeon-tau** | `filter@0.78 − filter@0.70` skip rate + AUC | Exploratory Wilcoxon | Skip rate, coverage AUC |
| **F-RQ-ceiling-discrete** (optional) | `hints − cma_me_bernoulli` and/or `cma_me − cma_me_bernoulli` @ 32.5k CA | Only if frozen in §10 before runs; else descriptive | Terminal coverage, mean fitness |

Confirmatory **F-B4-dungeon @ 5k remains primary** for journal abstract unless editors require relabelling; journal text must state “extended supplementary evidence.”

### 11.4 Quality gates (unchanged from §4–§5)

Re-check before C1: LLM fallback ≤5%, filter skip 25–45%, surrogate replay divergence ≤0.05. Stop C1 if projected calls exceed 100k **per stage** (run LLM arms sequentially if needed).

### 11.5 Execution order

1. Freeze §11 → §10 amendment row with date.  
2. C4 discrete CMA (CPU; retire ceiling encoding caveat early).  
3. C2 `random` (cheap sanity).  
4. C3 threshold ablation (CPU).  
5. C1 LLM marathon (`llm_stub` → `llm_hints` → `llm_hints_filter`, seeds 0–9).  
6. Stats script extensions + manuscript v2 + arXiv v2 same day as journal submit.

### 11.6 Manuscript / cover letter (journal v2)

> Extends arXiv:XXXX.XXXXX with (i) discrete Bernoulli/categorical CMA-ME on the CA grid (matched duel vs continuous relaxation ceiling), (ii) full-budget LLM dungeon factorial (32.5k proposals, $n{=}10$), (iii) `random` baseline, and (iv) filter threshold ablation. Confirmatory CA F-RQ-ceiling (continuous) and F-B4 @ 5k unchanged as primary; new results are supplementary families unless §10 freezes F-RQ-ceiling-discrete.

---

## 10. Amendment log

| Date | Change |
|------|--------|
| 2026-07-21 | **§8 Reviewer FAQ:** generality / budget asymmetry disclosure; rebuttal template; manuscript pointers. |
| 2026-07-21 | **§9 Optional extensions:** tiered compute plan + wall-time estimates from `q1-v4-dungeon-rerun` @ 5k. |
| 2026-07-22 | **Tier 0 DONE:** disclosure synced to `SWARM_EC_DRAFT_OUTLINE.md` §6.11, §7.4, rebuttal block, cover letter. |
| 2026-07-22 | **Tier 2 DONE:** `q1-v4-dungeon-full-cpu/` (20 runs @ 32.5k); stats family `v4-dungeon-cpu-full`; supplementary Holm PASS; manuscript Table `tab:b4-fullcpu`; arXiv v1 scope in Limitations. |
| 2026-07-22 | **§11 DRAFT:** journal extension package C (LLM @ 32.5k + random + τ ablation) — **not frozen**; amend §10 before first C-run. |
| 2026-07-22 | **§11 C4 DRAFT:** discrete Bernoulli/categorical CMA-ME on CA grid (retire continuous-relaxation caveat) — **not frozen**. |
| 2026-08-09 | **§7 AUC recipe wording + fig B4 no-LOCF (Reporting):** clarifies min-eval horizon / per-seed normalized AUC / no LOCF into Holm; regenerates appendix anytime plots. Locked `v4_dungeon_statistics.json` unchanged. |

