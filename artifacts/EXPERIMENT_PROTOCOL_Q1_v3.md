# Q1 experiment protocol v3 (baselines + multi-LLM + QD methods)

**Status:** §§1–4 **FROZEN** as of **2026-07-12**. §§5–12 remain operational drafts (tiers, cost, checklist execution).  
**Freeze rule:** changes to §§1–4 require a dated row in §12 + dual-report of pre/post wording (same bar as v2 RQ3 NI amendment). Do not start B1/B2/G1 runs under a silent edit.  
**External timestamp:** Zenodo integrity snapshot **10.5281/zenodo.21727011** (published **2026-07-31**; git tag `journal-v1` / `6d07dcdecf64`) — *not* a prospective pre-registration of completed runs; protocols remain frozen per family in this file + v2/v4/v5.  
**Extends:** [`EXPERIMENT_PROTOCOL_Q1_v2.md`](EXPERIMENT_PROTOCOL_Q1_v2.md) (grid primary + CVT sensitivity; results already in [`Q1_GRID_CVT_ANALYSIS.md`](Q1_GRID_CVT_ANALYSIS.md)).  
**Does not supersede v2 claims** until v3 confirmatory family is complete; v2 remains the frozen record of the first matrix.

Related: [`docs/SURROGATE_MODEL.md`](../docs/SURROGATE_MODEL.md), [`docs/MAPELITES.md`](../docs/MAPELITES.md), pyribs docs (CMA-ME / CMA-MAE).

---

## 0. Design overview

```text
v2 (DONE) ──► frozen: RQ1–RQ3 grid + CVT-s, D1 live exploratory, dual-report RQ3
                 │
v3 (THIS DOC) ──► publication-critical baselines + method comparisons
                 │
                 ├─ B1  Vanilla MAP-Elites (random emitter)     [CRITICAL]
                 ├─ B2  CMA-ME / CMA-MAE (pyribs)               [CRITICAL]
                 ├─ B3  Standard benchmarks (sphere + rastrigin) [SUPPLEMENTARY]
                 ├─ G1  RQ1 on ≥2 additional LLMs               [CRITICAL]
                 ├─ M1  MC-dropout vs GP+UCB (text / ablation)  [DONE]
                 └─ R1  SAIL / DSA-ME related-work paragraph     [DONE]
```

| Block | Priority | Role | Depends on new compute? |
|-------|----------|------|-------------------------|
| **B1** Vanilla ME (random) | critical for publication | Non-LLM / non-surrogate QD floor | Yes (CPU sims) |
| **B2** CMA-ME / CMA-MAE | critical for publication | Strong QD baseline (pyribs) | Yes (CPU; pyribs) |
| **B3** Sphere + rastrigin (pyribs) | supplementary (journal) | Standard-benchmark sanity; CMA + ME random | Yes (CPU; cheap) |
| **G1** Multi-LLM RQ1 | critical (generalizability) | Show hints effect ≠ one API model | Yes (LLM) |
| **M1** MC-dropout vs GP+UCB | medium | Justify surrogate+acquisition choice | Text and/or small ablation |
| **R1** SAIL / DSA-ME | cheap | Related-work positioning | Text only |
| **Stats lock** | critical | Hypotheses, Holm family, D1@0.45 | No (protocol only) |

**Inheritance from v2 (do not re-run unless invalidating change):**

- Grid `q1-full` / CVT `q1-cvt` matrices, checkpoint `nightly_v3_mc_d005`, stack v2 window **2026-07-02…2026-07-11**.
- RQ3 remains **exploratory** under v2 D1 (live div@0.45=0.528). v3 may reopen confirmatory RQ3 only if D1 passes under the **locked** v3 gate (§4).

---

## 1. Research questions (v3 delta) — FROZEN 2026-07-12

### 1.1 Carried from v2 (frozen wording)

| ID | Question | v2 status |
|----|----------|-----------|
| RQ1 | Surrogate hints vs stub | **PASS** (confirmatory) |
| RQ2 | Surrogate hold-out quality | Documented; R²≈0.94 composed @ gate 0.95 (aligned 2026-07-28) |
| RQ3 | Filter eval↓ without QD loss | **EXPLORATORY** (D1 fail; operational NI PASS) |
| RQ1-s / RQ3-s | CVT sensitivity | Sign-consistent; thresholds grid-transferred |

### 1.2 New v3 questions

| ID | Priority | Question | Conditions / comparison |
|----|----------|----------|-------------------------|
| **RQ0** | critical | Is LLM+surrogate competitive with **vanilla MAP-Elites** (random emitter only)? | `vanilla` vs v2 `stub` / `hints` (same budget, same archive) |
| **RQ4** | critical | How do we compare to **CMA-ME** and **CMA-MAE** (pyribs)? | `cma_me`, `cma_mae` vs `hints` (and optionally `vanilla`) |
| **RQ1-g** | critical | Does RQ1 **generalize** across LLM providers/models? | `stub` vs `hints` on ≥2 LLMs beyond `qwen-turbo` |
| **RQ5** | medium | Is MC-dropout + threshold_gate justified vs **GP+UCB**? | Text rationale and/or offline/online ablation |
| **RW** | cheap | Where do **SAIL / DSA-ME** sit relative to this stack? | Related-work only |

**Paper claim gate:** B1 + B2 + G1 required before camera-ready “QD + LLM + surrogate” story. M1/RW strengthen novelty framing, not primary tables.

---

## 2. Priority backlog (execution order) — FROZEN 2026-07-12

Lock this order before spending budget:

| # | Item | Priority | Exit criterion |
|---|------|----------|----------------|
| 1 | Freeze §§1–4 + D1@0.45 + hypothesis list | critical | **DONE 2026-07-12** — further edits need §12 amendment |
| 2 | **B1** Vanilla MAP-Elites (random emitter) | critical | **DONE 2026-07-12** — n=10; F-RQ0 PASS |
| 3 | **B2** CMA-ME + CMA-MAE (pyribs) | critical | **DONE 2026-07-13** — 20/20 runs; F-RQ4 FAIL |
| 4 | **G1** RQ1 on ≥2 extra LLMs | critical | **DONE 2026-07-22** — both providers **n=10 F-RQ1g PASS** (gpt-4o-mini 2026-07-21; DeepSeek 2026-07-22) |
| 5 | **M1** MC-dropout vs GP+UCB write-up (± ablation) | medium | **DONE 2026-07-22** — Phases 0–4; paragraph in `draft_v0.tex` §acquisition + Limitations; `gp_ucb_ablation.json` |
| 6 | **R1** SAIL / DSA-ME related-work | cheap | **DONE 2026-07-22** — expanded `draft_v0.tex` §2.2 + positioning note; protocol §8 filled |
| 7 | **B3** Standard QD benchmarks (sphere + rastrigin) | supplementary | **DONE 2026-07-17** — implementation + 50/50 runs; descriptive sanity PASS |
| 8 | Optional: reopen RQ3 confirmatory if D1@0.45 passes | conditional | **SUPERSEDED 2026-07-28** — see §4.2.1 F-RQ3-gray |

---

## 3. Method blocks — FROZEN 2026-07-12 (specs wired; remaining ops in §9)

### 3.1 B1 — Vanilla MAP-Elites (random emitter) *[critical]*

**Intent:** publication-critical non-LLM baseline. Shows gains of stub/hints/filter over classic random variation + archive.

| Knob | Locked value |
|------|----------------------|
| Archive | Grid 50×50 (primary); optional CVT later |
| Emitters | **random only** (no genetic, no LLM) |
| Iterations / batch | Match v2 nightly: **650 × 50** |
| Seeds | **0–9** (paired with v2) |
| Surrogate | `enabled: false` (no hints, no filter) |
| Output | `artifacts/experiments/q1-v3-vanilla/` |

**Scheduler:** `map_elites_scheduler_nightly_vanilla.yaml` — **50× random**; `llm.enabled: false`; surrogate off.

**Tier (complete 2026-07-12):** `q1-v3-vanilla` → `artifacts/experiments/q1-v3-vanilla/vanilla/seed_*/`

```bash
# Full RQ0 matrix (seeds 0–9): DONE 2026-07-12
./scripts/run_experiment_batch.sh q1-v3-vanilla
```

**Result (n=10, paired vs v2 `q1-full` hints):** mean Δcov(hints−vanilla) **+18.05 pp**, Δfit **+0.067** (occupied-bin mean; CSV `mean_best_fitness`), sign **10/10**; Wilcoxon p≈0.001; Holm m=2 PASS → **F-RQ0 PASS**. Details: [`q1-v3-vanilla/ANALYSIS.md`](experiments/q1-v3-vanilla/ANALYSIS.md). (Supersedes an early note that listed Δfit +0.166 / hints fit 0.539 from a mis-read fitness column.)

**Metrics:** coverage, mean_best_fitness, evaluations (= full budget), wall time. Compare paired Δ vs v2 `stub` and `hints`.

**Hypothesis (pre-registered):** median Δcov(`hints` − `vanilla`) > 0 (one-sided Wilcoxon, n=10); report bootstrap CI. Confirmatory family **F-RQ0** (§4): hints−vanilla Δcov+Δfit conjunctive, Holm m=2. **Observed: PASS.**

**Naming (2026-07-14):** label **`vanilla`** = **random-only floor** (50× random), not “canonical MAP-Elites” in the Mouret & Clune sense. Report as non-adaptive QD floor in the manuscript; see **§3.1b** for archive-variation baseline.

### 3.1b B1b — Genetic MAP-Elites (matched slots, no LLM) *[critical add-on]*

**Intent:** address reviewer concern that B1 random-only is too weak vs literature “vanilla ME”. Same nightly budget and slot proportions as stub/hints, but **10 LLM slots → genetic** (20 random + **30 genetic**); no surrogate, no API.

| Knob | Locked value |
|------|----------------------|
| Archive | Grid 50×50; same warm-start as v2/vanilla |
| Emitters | **20× random + 30× genetic** (uniform crossover + Gaussian mutation, `mutation_scale: 0.02`) |
| Iterations / batch | **650 × 50** |
| Seeds | **0–9** (paired with v2 hints) |
| LLM / surrogate | off |
| Output | `artifacts/experiments/q1-v3-genetic-me/` |

**Scheduler:** `map_elites_scheduler_nightly_genetic_me.yaml`

```bash
# Full matrix (seeds 0–9); CPU-only ~4 min/seed
./scripts/run_experiment_batch.sh q1-v3-genetic-me
```

**Reporting:** descriptive paired Δ vs v2 `hints` and vs B1 `vanilla`; **not** a pre-registered Holm family (amendment add-on). Expectation from levels: `vanilla` (~42%) < `stub` (~45%) ≲ `genetic_me` < `hints` (~60%) ≪ `cma_me` (~73%).

**Does not replace F-RQ0** (hints > random floor remains frozen); strengthens Discussion / Fig. 5 ladder.

### 3.2 B2 — CMA-ME / CMA-MAE via pyribs *[critical]*

**Intent:** strong QD baselines outside the LifeManifold emitter stack.

| Knob | Locked value |
|------|----------------------|
| Library | **pyribs** (`ribs` — pin in §9 / [`Q1_V3_B2_PYRIBS_TASKS.md`](Q1_V3_B2_PYRIBS_TASKS.md) Hyperparams) |
| Algorithms | **CMA-ME** and **CMA-MAE** (separate arms) |
| BC / fitness | Same stability–diversity BC and illuminator fitness as MAP-Elites |
| Eval budget | **Match** v2 total sims per seed (32 500) |
| Archive resolution | Grid 50×50 equivalent (pyribs `GridArchive` dims `(50,50)`, ranges `[(0,1),(0,1)]`) |
| Seeds | **0–9** |
| Warm-start | **Yes** — same grid baseline as RQ0/v2 ME: `artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl` |
| Genotype | Genetic **21-D** (`encode_world` / `decode_genome`); CMA proposes continuous θ; decode **rints** rule bits (documented limitation) |
| Surrogate / LLM | **Off** (wrapper only; not `EmitterKind`) |
| pyribs emitters | **5× `EvolutionStrategyEmitter`** (homogeneous CMA-ES pool; not mixed RDA/Improvement — §3.9) |
| Output | `artifacts/experiments/q1-v3-pyribs/{cma_me,cma_mae}/` |

**CMA hyperparameters (T0 lock — inline; also in run JSON `pyribs_hyperparams`):**

| Knob | CMA-ME | CMA-MAE |
|------|--------|---------|
| `ribs` version | **0.11.0** | same |
| **λ** (ask / iteration) | **5 × 50 = 250** | same |
| **# asks** | **130** → **32 500** evals | same |
| **σ₀** | **0.2** | **0.2** |
| **x₀** | mid-bounds: 18×`0.5` + noise `0.1` + regen `0.25` + pred `0.5` | same |
| **restart_rule** | **`no_improvement`** | **`basic`** |
| ranker | `2imp` | `imp` |
| selection_rule | `filter` | `mu` |
| learning_rate | 1.0 | 0.01 |
| result_archive | no (primary) | yes (report coverage/fit from result) |
| ES box bounds | **off** (clip/rint at decode; avoids σ₀ resample storms) | same |

**Integration:** wrapper `scripts/run_pyribs_baseline.py` writes nightly-compatible
summary columns. Current runner emits a `pyribs_hyperparams` block for new runs;
the frozen B2 summaries predate that block, so their lock is carried by the
§3.2 table and [`Q1_V3_B2_PYRIBS_TASKS.md`](Q1_V3_B2_PYRIBS_TASKS.md).

**Hypotheses (pre-registered) — aligned with §4.1 F-RQ4:**

- For each arm (`cma_me`, `cma_mae`): paired Δcov and Δfit of **`hints` − arm**, both **one-sided greater**, conjunctive; Holm family **m=4**.
- Do **not** claim LLM superiority unless Holm family passes.
- ~~two-sided or TOST — choose before runs~~ (**superseded 2026-07-12**; see §12).

**Result (2026-07-13, T4):** **F-RQ4 FAIL** — family does not support hints > both CMA arms. Observed: CMA-ME ahead on coverage (mean Δcov hints−me **−12.96** pp); CMA-MAE near parity on cov, hints ahead on fit (+0.050). Details: [`q1-v3-pyribs/ANALYSIS.md`](experiments/q1-v3-pyribs/ANALYSIS.md). (§12 amendment; frozen hypothesis text above unchanged.)

**Risk:** BC scaling / fitness composition must match illuminator exactly (incl. early_extinct). Document adapter tests. Bit-rounding on 18 rule genes means CMA-ES explores a continuous relaxation of the genetic genotype.

### 3.3 G1 — RQ1 on ≥2 additional LLMs *[critical]*

**Intent:** generalizability beyond floating `qwen-turbo` (v2 window-pin).

| Knob | Locked value |
|------|----------------------|
| Reference | v2 `qwen-turbo` RQ1 (already done) — **do not re-run** unless stack changes |
| New models | **≥2** distinct APIs/models (propose before runs; examples: dated `qwen-turbo-YYYY-MM-DD` or `qwen-flash-…`, plus one non-Qwen e.g. OpenAI mini / other) |
| **G1 pick (2026-07-12)** | **(1)** `deepseek-v4-pro` @ DeepSeek official · **(2)** `gpt-4o-mini` @ OpenAI (wired) |
| Pinning | **Dated / explicit model IDs**; log request+response `model`, `api_base`, run window |
| Arms | Per model: `stub` + `hints` only (filter out of G1 scope unless budget allows) |
| Seeds | Prefer **0–9**; minimum **0–4** if budget-capped (document power loss) |
| Prompt / stack | Same v2 prompts (`world_spec` only); same `temperature=0.2`; DeepSeek **`thinking: {type: disabled}`** |
| Output | `artifacts/experiments/q1-v3-llm/<model_slug>/` |

**DeepSeek V4 Pro — n=10 complete (2026-07-22)**

```bash
export DEEPSEEK_API_KEY=...
# G1 minimal (seeds 0–4): DONE 2026-07-12
./scripts/run_experiment_batch.sh q1-v3-llm-deepseek-v4-pro
# Full n=10: DONE 2026-07-22 (seeds 5–9 2026-07-21 → 2026-07-22)
./scripts/run_experiment_batch.sh q1-v3-llm-deepseek-v4-pro 0 9
```

**API / model pin (logged uniformly on all 20 runs):**

| Field | Pinned value |
|-------|----------------|
| `model` | `deepseek-v4-pro` |
| `api_base` | `https://api.deepseek.com/v1/chat/completions` |
| Thinking | `thinking: {type: disabled}` |
| `llm_spec_hash` | `fd5f0d34f97f9df7` |
| Spec | `worldspace/specs/llm_world_generator_deepseek.yaml` |
| Window seeds 0–4 | **2026-07-12 03:11 → 18:54** (local) |
| Window seeds 5–9 | **2026-07-21 14:10 → 2026-07-22 09:57** |
| Fallback | **0%** |

**G1 result (n=10):** mean Δcov **+16.95 pp**, Δfit **+0.052**, sign **10/10** cov · **10/10** fit; Wilcoxon p_cov=**0.00098**, p_fit=**0.00098**; Holm m=2 **both reject** → **F-RQ1g PASS** for this provider. Absolute hints **61.93%** (≈ qwen **60.41%**). Details: [`deepseek-v4-pro/ANALYSIS.md`](experiments/q1-v3-llm/deepseek-v4-pro/ANALYSIS.md), `frq1g_statistics.json`.

**G1-minimal result (n=5, historical):** mean Δcov **+16.46 pp**, mean Δfit **+0.050**, sign **5/5** both; Wilcoxon 1s p=0.031.

**OpenAI gpt-4o-mini — n=10 complete (2026-07-21)**

```bash
export OPENAI_API_KEY=...
# G1 minimal (seeds 0–4): DONE 2026-07-14
./scripts/run_experiment_batch.sh q1-v3-llm-gpt-4o-mini
# Full n=10: DONE 2026-07-21 (seeds 5–9 nohup 2026-07-20 → 2026-07-21)
./scripts/run_experiment_batch.sh q1-v3-llm-gpt-4o-mini 0 9
```

**API / model pin:**

| Field | Pinned value |
|-------|----------------|
| Request `model` | `gpt-4o-mini` |
| Response `model` (smoke) | `gpt-4o-mini-2024-07-18` |
| `api_base` | `https://api.openai.com/v1/chat/completions` |
| Spec | `worldspace/specs/llm_world_generator_openai.yaml` |
| `llm_spec_hash` | `b678837d32da4f49` |
| Tier | `q1-v3-llm-gpt-4o-mini` → `artifacts/experiments/q1-v3-llm/gpt-4o-mini/` |
| Window seeds 0–4 | **2026-07-13 03:48 → 2026-07-14 17:58** |
| Window seeds 5–9 | **2026-07-20 → 2026-07-21** |
| Fallback | max **0.26%** (hints/seed_5); mean **0.04%** |

**G1 result (n=10):** mean Δcov **+9.66 pp**, Δfit **+0.018**, sign **10/10** cov · **9/10** fit; Wilcoxon p_cov=**0.0010**, p_fit=**0.0029**; Holm m=2 **both reject** → **F-RQ1g PASS** for this provider. Absolute hints **55.34%** vs qwen/DeepSeek **~60%**. Details: [`gpt-4o-mini/ANALYSIS.md`](experiments/q1-v3-llm/gpt-4o-mini/ANALYSIS.md), `frq1g_statistics.json`.

**G1-minimal result (n=5, historical):** mean Δcov **+11.09 pp**, mean Δfit **+0.070**, sign **5/5** both; Wilcoxon 1s p≈0.031.

**G1 reporting:** Both additional LLMs are **confirmatory F-RQ1g PASS @ n=10** (per provider; never pooled). Primary bundled F-RQ1 remains v2 qwen (n=10). gpt-4o-mini absolute hints lag qwen/DeepSeek (~55% vs ~60–62%).

**Hypothesis (pre-registered):** for each new LLM, RQ1 conjunctive (Δcov↑ and Δfit↑) with same local_ok / Holm rules as v2 §7, or a declared multi-LLM Holm family (§4).

### 3.4 M1 — MC-dropout vs GP+UCB *[medium]*

**Intent:** justify why acquisition is MLP+MC-dropout+`threshold_gate`, not classic SAIL-style GP+UCB.

| Path | Cost | Deliverable |
|------|------|-------------|
| **Text-only (minimum)** | cheap | §7 rationale: buffer size, online cost, Q1 fitness-gate design (`max_u=1.0`), citation to SAIL/UCB |
| **Ablation (stretch)** | medium | Offline: GP on hold-out subset vs MLP components; and/or 1-seed shadow comparing UCB skip vs threshold_gate |

**Exit:** reviewer-facing paragraph + optional `artifacts/surrogate/gp_ucb_ablation.json`. Not required for B1/B2/G1 tables.

**Progress (2026-07-22):** Phase 0–4 **DONE** — [`M1_PLAN.md`](surrogate/M1_PLAN.md), [`M1_ANALYSIS.md`](surrogate/M1_ANALYSIS.md), `gp_ucb_ablation.json`; paragraph pasted into [`draft_v0.tex`](manuscript/draft_v0.tex) §acquisition + Limitations. MLP R² **0.761** vs GP **0.223** (~12× faster). Offline: `threshold_gate`+MLP false-skip **0.7%** vs GP+UCB ~**3%**. Live counterfactual (n=325k): skip **0.335→0.285/0.257/0.212** under UCB on logged MLP μ/σ.

### 3.5 R1 — SAIL / DSA-ME related work *[cheap]*

**Intent:** position LifeManifold stack without new experiments.

**Status:** **DONE 2026-07-22** — expanded in [`draft_v0.tex`](manuscript/draft_v0.tex) §2.2 (Surrogate-assisted illumination) + table positioning note; filled §8 below. Cross-links M1 ablation and D1 disclosure.

Landed points:

- **SAIL** — GP + acquisition over BC/fitness; we use MLP components + composed fitness + optional `threshold_gate`.
- **DSA-ME** — online neural inner loop / outer validation; contrast discrete CA genome + offline shared checkpoint + LLM emitter.
- **Ours** — before/after attach framework; stochastic seeded CA; D1 compose-gate mismatch disclosed; pyribs as QD baselines (not SAIL re-implementation).

No runs.

### 3.6 Target-selection parity (stub vs hints) *[disclosure + post-arXiv sensitivity]*

**Confound (locked historical runs):** v2 RQ1 and v3 G1 compare `stub` vs `hints` under the same nightly slot mix, but **target selection differs**:

| Arm | YAML | `target_selection` | Surrogate |
|-----|------|--------------------|-----------|
| `stub` | `map_elites_scheduler_nightly_llm_stub.yaml` | **`min_fitness_frontier`** (scheduler default) | off |
| `hints` / `filter` | `map_elites_scheduler_nightly_llm.yaml`, `_filter.yaml`, … | **`uniform_frontier`** | on (+ gate / acquisition) |

`min_fitness_frontier` sends all LLM batch slots at the archive cell with minimum fitness on the frontier; `uniform_frontier` spreads slots across frontier cells (see filter YAML comment). The RQ1 effect is therefore a **bundled** comparison (surrogate hints **and** target-selection policy), not an isolated surrogate ablation.

**arXiv reporting (2026-07-14):** disclose in Methods/Limitations; soften causal language (“bundled nightly arms”). **Do not** mutate existing `stub` YAML or re-run v2 / G1 matrices.

**Matched sensitivity (completed 2026-07-17; descriptive):**

| Knob | Value |
|------|-------|
| Scheduler | `map_elites_scheduler_nightly_llm_stub_uniform.yaml` — stub + `uniform_frontier`, surrogate off |
| Condition label | `stub_uniform` |
| Tier | `q1-stub-uniform-sensitivity` → `artifacts/experiments/q1-stub-uniform-sensitivity/` |
| Seeds | **0–9 DONE** (2026-07-19); compare vs frozen `hints` on same seeds |
| Exit | Descriptive Δ(`hints` − `stub_uniform`) on matched target selection; if ≈ v2 Δ(`hints` − `stub`), surrogate signal dominates; if gap shrinks, disclose residual confound |

```bash
# Completed sensitivity (full n=10):
./scripts/run_experiment_batch.sh q1-stub-uniform-sensitivity 0 9
```

**Result (seeds 0–9):** mean Δcov(`hints` − `stub_uniform`) **+0.26 pp** (60.41% vs 60.15%); mean Δfit **+0.0033**; Wilcoxon paired **p = 0.41** (exploratory; not confirmatory NULL). Per-seed Δcov: −1.12, −0.76, +1.44, +2.00, +1.28, −0.76, +0.56, −0.92, +1.40, −0.52. Subsample: seeds 0–4 **+0.57 pp**; seeds 5–9 **−0.02 pp**. By contrast, mean Δcov(`hints` − `stub`) **+15.09 pp** and (`stub_uniform` − `stub`) **+14.83 pp** (both Wilcoxon **p = 0.002**). **Interpretation:** at matched `uniform_frontier`, scalar surrogate hints remain **descriptively flat** and do **not** explain the historical +15 pp bundled gap; target selection is the dominant confound. Analysis: [`q1-stub-uniform-sensitivity/ANALYSIS.md`](experiments/q1-stub-uniform-sensitivity/ANALYSIS.md).

### 3.6b Component-rich hints pilot (RQ1b / Path 4A) *[descriptive; 2026-07-15]*

**Intent:** test whether **7 Strategy-A surrogate components** in the LLM user prompt (not only `{surrogate_mean}` / `{surrogate_uncertainty}`) recover QD when `target_selection` is matched.

| Knob | Value |
|------|-------|
| Scheduler | `map_elites_scheduler_nightly_llm_hints_rich.yaml` |
| User prompt | `prompts/map_elites_llm_emitter_user_components.txt` |
| Condition | `hints_rich` |
| Tier | `q1-hints-rich-pilot` → `artifacts/experiments/q1-hints-rich-pilot/` |
| Match vs `hints` | `uniform_frontier`, same checkpoint `nightly_v3_mc_d005`, same slot mix |
| Seeds | Pilot **0** only (extend 1–4 only if Δcov ≥ +2 pp vs `hints` or `stub_uniform`) |

```bash
./scripts/run_experiment_batch.sh q1-hints-rich-pilot 0 0
```

**Result (seed 0, 2026-07-15):**

| Arm | Cov % | Mean best fit | User prompt hash |
|-----|-------|---------------|------------------|
| `hints` (frozen) | 59.72 | 0.4981 | `e2afd1e9` |
| `stub_uniform` | 60.84 | 0.4956 | `e2afd1e9` |
| **`hints_rich`** | **60.12** | **0.4958** | **`39effe9f`** ✓ |

Δcov: hints_rich − hints **+0.40 pp**; hints_rich − stub_uniform **−0.72 pp**. Δfit: hints_rich − hints **−0.0023**. **Verdict: NULL** — rich components did not pass pilot thresholds (+2 pp / +0.02 fit). **Do not** extend seeds 1–4. Analysis: [`q1-hints-rich-pilot/ANALYSIS.md`](experiments/q1-hints-rich-pilot/ANALYSIS.md).

**Reporting:** descriptive only; not a Holm family. Supports manuscript decomposition: **hint channel content (scalar or rich) is not a measurable QD driver** once `uniform_frontier` is held fixed; historical RQ1 PASS remains a **bundled** nightly-arm comparison.

### 3.6c Parent-metrics hint header pilot (RQ1d / Path 4D) *[descriptive; 2026-07-15]*

**Intent:** test whether **observed parent-cell simulation metrics** in the **hint header** (above `current_elite_json`, metrics not duplicated in JSON block) improve QD vs scalar/component surrogate hints when `target_selection` is matched.

| Knob | Value |
|------|-------|
| Scheduler | `map_elites_scheduler_nightly_llm_hints_parent.yaml` |
| User prompt | `prompts/map_elites_llm_emitter_user_parent_hints.txt` |
| Condition | `hints_parent` |
| Tier | `q1-hints-parent-pilot` → `artifacts/experiments/q1-hints-parent-pilot/` |
| Pre-flight | `scripts/preflight_hints_parent.py` — GO if median \|parent − surrogate fitness\| ≥ 0.02 on archive elites |
| Seeds | Pilot **0** only |

```bash
uv run python scripts/preflight_hints_parent.py   # gate before run
./scripts/run_experiment_batch.sh q1-hints-parent-pilot 0 0
```

**Pre-flight (2026-07-15):** on collapsed `q1-full/hints/seed_0` archive, median \|Δfit\| **0.127** (n=1184) → **GO** (numbers differ materially).

**Result (seed 0, 2026-07-15):**

| Arm | Cov % | Mean best fit | User prompt hash |
|-----|-------|---------------|------------------|
| `hints` (frozen) | 59.72 | 0.4981 | `e2afd1e9` |
| `stub_uniform` | 60.84 | 0.4956 | `e2afd1e9` |
| `hints_rich` | 60.12 | 0.4958 | `39effe9f` |
| **`hints_parent`** | **59.92** | **0.4945** | **`6197d5e2`** ✓ |

Δcov: hints_parent − hints **+0.20 pp**; hints_parent − stub_uniform **−0.92 pp**; hints_parent − hints_rich **−0.20 pp**. Δfit: hints_parent − hints **−0.0036**. **Verdict: NULL** — observed parent header did not pass pilot thresholds (+2 pp / +0.02 fit) despite discriminative pre-flight. **Do not** extend seeds 1–4. Analysis: [`q1-hints-parent-pilot/ANALYSIS.md`](experiments/q1-hints-parent-pilot/ANALYSIS.md).

**Reporting:** closes Path 4 prompt-engineering line (4A scalar-rich, 4D parent header); hint **salience/format** is not a QD driver at matched `uniform_frontier`.

### 3.6d Direction-of-improvement hints pilot (RQ1e / Path 5E) *[descriptive; 2026-07-15]*

**Intent:** test whether **actionable local edit hints** (finite-difference ∂fit/∂feature → rule toggles / param edits) improve QD vs scalar hints when `target_selection` is matched — last meaningful prompt-format test before closing the hint channel.

| Knob | Value |
|------|-------|
| Scheduler | `map_elites_scheduler_nightly_llm_hints_direction.yaml` |
| User prompt | `prompts/map_elites_llm_emitter_user_direction.txt` |
| Condition | `hints_direction` |
| Tier | `q1-hints-direction-pilot` → `artifacts/experiments/q1-hints-direction-pilot/` |
| Pre-flight | `scripts/preflight_hints_direction.py` — GO if median max \|∂fit/∂x\| ≥ 0.01 on archive elites |
| Seeds | Pilot **0** only |

```bash
uv run python scripts/preflight_hints_direction.py   # gate before run
./scripts/run_experiment_batch.sh q1-hints-direction-pilot 0 0
```

**Pre-flight (2026-07-15):** on collapsed `q1-full/hints/seed_0` archive, median max \|∂fit/∂x\| **0.783** (n=1184) → **GO** (gradients strong enough to be actionable in prompt).

**Result (seed 0, 2026-07-15):**

| Arm | Cov % | Mean best fit | User prompt hash |
|-----|-------|---------------|------------------|
| `hints` (frozen) | 59.72 | 0.4981 | `e2afd1e9` |
| `stub_uniform` | 60.84 | 0.4956 | `e2afd1e9` |
| `hints_rich` | 60.12 | 0.4958 | `39effe9f` |
| `hints_parent` | 59.92 | 0.4945 | `6197d5e2` |
| **`hints_direction`** | **59.04** | **0.5040** | **`1ebbb172`** ✓ |

Δcov: hints_direction − hints **−0.68 pp**; hints_direction − stub_uniform **−1.80 pp**; hints_direction − hints_parent **−0.88 pp**. Δfit: hints_direction − hints **+0.0059**. **Verdict: NULL** — direction hints did not pass pilot thresholds (+2 pp / +0.02 fit) despite strong pre-flight gradients and 0% LLM fallback. **Do not** extend seeds 1–4 or shuffled-direction control. Analysis: [`q1-hints-direction-pilot/ANALYSIS.md`](experiments/q1-hints-direction-pilot/ANALYSIS.md).

**Reporting:** **hint channel closed** for qwen + nightly v3 — scalar, component-rich, parent-header, direction-of-improvement, and **weak omni-7b** formats all ≈ NULL within the ~60% / ~57% / ~0.50 bands at matched `uniform_frontier`.

### 3.6e Weak-model × hints interaction (RQ1f / G2) *[descriptive; post-arXiv]*

**Intent:** test moderation hypothesis — **weaker** LLMs may benefit more from explicit `{surrogate_mean}` / `{surrogate_uncertainty}` than capable models where hint **content** is already NULL at matched `uniform_frontier` (§3.6–§3.6d).

| Knob | Value |
|------|-------|
| LLM spec | `llm_world_generator_weak.yaml` → **`qwen2.5-omni-7b`** @ DashScope intl (`QWEN_API_KEY`; `qwen2.5-7b-instruct` **403** on this workspace) |
| Scheduler (hints) | `map_elites_scheduler_nightly_llm_weak_hints.yaml` |
| Scheduler (control) | `map_elites_scheduler_nightly_llm_stub_uniform.yaml` (reuse) |
| Tier | `q1-v3-llm-weak-pilot` → `artifacts/experiments/q1-v3-llm-weak-pilot/` |
| Arms | `stub_uniform` + `hints` (both `@ uniform_frontier`) |
| Pre-flight | Nominal plan: ≥95% on 50 calls; executed budget smoke **5/5 parse** (documented deviation) |
| Seeds | Pilot **0–2** (interaction vs pre-registered frozen n=3 strong Δcov **−0.15 pp**; later strong n=5 sensitivity is +0.57 pp) |

```bash
uv run python scripts/preflight_llm_weak.py
./scripts/run_experiment_batch.sh q1-v3-llm-weak-pilot
```

**Primary endpoint (exploratory):** Δcov\_weak = mean(hints\_weak) − mean(stub\_uniform\_weak) vs the pre-registered frozen Δcov\_strong = −0.15 pp (qwen, seeds 0–2; retained for endpoint integrity despite later n=5 completion). **PASS** if Δcov\_weak ≥ **+2 pp** **and** (Δcov\_weak − Δcov\_strong) ≥ **+1 pp**.

**Not primary:** G1 bundled stub (`min_fitness_frontier`) vs hints — confounded; gpt-4o-mini already shows **smaller** bundled gap (+11 pp vs +16 pp), not larger.

**Status (2026-07-16):** seeds **0–2 DONE** — **NULL**. Mean Δcov\_weak(hints−stub\_uniform) **−0.08 pp**; interaction vs strong Δcov **+0.07 pp** (need ≥ +1 pp). See [`q1-v3-llm-weak-pilot/ANALYSIS.md`](experiments/q1-v3-llm-weak-pilot/ANALYSIS.md). **Hint channel closed** for weak model (omni-7b) as well.

### 3.7 Anytime archive trace (eval-indexed curves) *[wired; matched acquisition traces complete]*

**Intent:** support anytime QD plots (coverage / mean best fitness vs evaluation budget) without changing confirmatory primary endpoints (final metrics @ 32 500 evals).

| Knob | Value |
|------|-------|
| Artifact | `archive_trace.jsonl` per run directory |
| MAP-Elites | one row per iteration when `performance.log_iteration_timing: true` (all nightly YAMLs); includes **iteration 0** warm-start snapshot |
| pyribs B2 | one row per CMA ask (+ ask 0 warm-start) |
| Fields | `evaluations`, `filled_cells`, `coverage` (0–1), `mean_best_fitness` |
| Plot helper | `scripts/plot_anytime_qd.py` |

**Frozen v2/v3/G1 matrix:** original runs lack traces. New matched acquisition
runs (`genetic_me_uniform`, `genetic_me_filter`, n=10 each) include complete
traces and support the §3.8 convergence analysis.

**Performance ladder traces (2026-07-20 DONE):** tier `q1-anytime-ladder` →
`artifacts/experiments/q1-anytime-ladder/`; seeds **0–4** × arms `vanilla`,
`hints`, `cma_me` (650 iters, 32 500 evals). Does **not** overwrite frozen
`q1-full` / `q1-v3-pyribs`. Terminal parity vs frozen: vanilla and cma_me
**bit-identical**; hints **+0.19 pp** mean (±2 pp seed noise). Anytime
medians @ 10k / 20k / 32.5k evals: vanilla **40.0 / 41.0 / 42.2%**; hints
**47.2 / 54.0 / 60.0%**; cma_me **60.9 / 65.4 / 71.4%** (n=5, IQR in
ANALYSIS.md). Manuscript Fig. 8 (`fig08_anytime_ladder.pdf`).

**Reporting:** descriptive / appendix only; x-axis = **evaluations**, not wall-clock (LLM latency unfair).

### 3.7b CMA encoding ablation (F-RQ-ceiling honesty) *[DONE 2026-07-20]*

**Intent:** test whether the CMA-ME coverage ceiling is robust to rule-bit decode and warm-start (reviewer attack on continuous-relaxation fairness).

| Knob | Value |
|------|-------|
| Tier | `q1-cma-encoding-ablation` → `artifacts/experiments/q1-cma-encoding-ablation/` |
| Algo | CMA-ME only; frozen reference = `q1-v3-pyribs/cma_me` (`decode_mode=rint`, warm-start) |
| Arms | `cma_me_threshold`, `cma_me_bernoulli`, `cma_me_cold` (no warm-start) |
| Seeds | **0–4 DONE** (n=5) |

**Results (mean coverage, seeds 0–4):** threshold **72.64%** (Δ vs rint **0.00 pp**, bit-identical all seeds); bernoulli **72.18%** (−0.46 pp); cold **69.17%** (−3.47 pp). Weakest arm still **+8.8 pp** vs hints (~60%). **Verdict:** ceiling direction stable; F-RQ-ceiling caveat holds with data.

Analysis: [`q1-cma-encoding-ablation/ANALYSIS.md`](experiments/q1-cma-encoding-ablation/ANALYSIS.md)

### 3.8 Factorial ablation (−LLM, +surrogate filter) *[DONE 2026-07-17; descriptive]*

**Intent:** close the missing **LLM×surrogate 2×2** cell: surrogate-guided MAP-Elites **without** LLM (threshold filter on genetic/random emitters only). Literal “surrogate hints without LLM” is impossible—hints are an LLM prompt channel; this arm tests **acquisition filter** instead.

| Cell | Arm | LLM | Surrogate | Notes |
|------|-----|-----|-----------|-------|
| (−, −) | `genetic_me` | off | off | B1b; historical `min_fitness_frontier` |
| (−, −), matched | **`genetic_me_uniform`** | off | off | `uniform_frontier`; clean filter control |
| (−, +) | **`genetic_me_filter`** | off | on + filter | **this section** |
| (+, −) | `stub` | on | off (stub constants in prompt) | RQ1 control |
| (+, +) | `hints` | on | on + hints gate | RQ1 treatment |
| (+, +, skip) | `filter` | on | on + filter | RQ3 exploratory; same filter policy, LLM slots kept |

| Knob | Locked value |
|------|----------------|
| Emitters | **20× random + 30× genetic** (same as `genetic_me`; no LLM slots) |
| Filter policy | Same as `q1-full` filter: `threshold_gate`, `min_predicted_fitness=0.45`, `never_skip_empty_bin=true` |
| `target_selection` | `uniform_frontier` (aligned with hints/filter) |
| Output | `artifacts/experiments/q1-v3-genetic-me-filter/` |

**Schedulers:** `map_elites_scheduler_nightly_genetic_me_filter.yaml`; matched control `map_elites_scheduler_nightly_genetic_me_uniform.yaml`

```bash
# Matched no-filter control (CPU-only; default seeds 0–9):
./scripts/run_experiment_batch.sh q1-v3-genetic-me-uniform

# Filter arm (complete; eval count < 32 500 due to skips):
./scripts/run_experiment_batch.sh q1-v3-genetic-me-filter
```

**Reporting:** descriptive factorial table vs `genetic_me`, `stub`, `hints`; **not** a new Holm family. Primary surrogate ablation for the LLM channel remains **stub vs hints** (RQ1). CMA baselines compare **full pipelines**, not factorial cells.

**Status (2026-07-17):** matched `genetic_me_uniform` and `genetic_me_filter`
seeds **0–9 DONE**. At fixed 650 iterations, filter saves **33.5%** simulator
evaluations (21.6k vs 32.5k) at Δcov **−0.83 pp** and Δfit **−0.0015**. At
equal evaluation budget, filter leads by **+2.96 pp @ 10k**, **+3.72 pp @ 15k**,
and **+3.65 pp @ 20k** (10/10). It reaches 50% / 55% coverage with **28% /
29% fewer evaluations**. Wall time is worse (**19.9 vs 6.3 min**) because MLP
inference exceeds the cost of skipped CA simulations. **Verdict: acquisition
channel OPEN for sample efficiency, not wall-clock speed.** See
[`q1-v3-genetic-me-uniform/ANALYSIS.md`](experiments/q1-v3-genetic-me-uniform/ANALYSIS.md)
and [`q1-v3-genetic-me-filter/ANALYSIS.md`](experiments/q1-v3-genetic-me-filter/ANALYSIS.md).

**Does not replace** RQ1 or amend F-RQ0/F-RQ4.

#### Reviewer FAQ: factorial vs CMA *(disclosure; arXiv text)*

**Q: Why not a full LLM×surrogate 2×2 against CMA-ME/MAE?**

| Issue | Answer |
|-------|--------|
| **Confound** | `hints` bundles **LLM emitters + surrogate prompt hints** (+ `uniform_frontier`); pyribs arms bundle **neither** — cross-pipeline comparison, not same factorial grid. |
| **Ill-defined CMA cells** | “CMA + LLM” / “CMA + surrogate filter” are **different architectures**, not factor levels of the LifeManifold stack; not pre-registered and out of B2 scope. |
| **What we *do* factorialize** | **Within LifeManifold**: matched `genetic_me_uniform` (−LLM/−surrogate) vs `genetic_me_filter` (−LLM/+filter), both at `uniform_frontier`; `stub_uniform` vs `hints` for the LLM-on hint channel. |
| **Frozen evidence today** | **RQ1** hint channel is descriptively flat at matched target selection (§3.6, n=10). Acquisition filter improves evaluation-indexed convergence at matched `uniform_frontier`. |
| **Matched (−,+) contrast** | `genetic_me_uniform` vs `genetic_me_filter`, n=10: +3.65 pp filter @ 20k eval; −0.83 pp filter at fixed 650 iterations with 33.5% fewer sims. |
| **vs CMA interpretation** | F-RQ4 compares **end-to-end QD** at matched 32 500 evals; does **not** attribute CMA−hints gap to “LLM weak” vs “surrogate harmful” — use internal factorial + performance **ladder** (vanilla → genetic_me → stub/hints → cma_me). |
| **arXiv / journal policy** | Report descriptively; no Holm amendment. Historical +12.28 pp remains target-selection-confounded; use matched anytime result for acquisition claims. |

**Suggested rebuttal sentence:** *We pre-specified a LifeManifold 2×2-style ablation (§3.8); RQ1 isolates the surrogate hint channel; CMA baselines are strong full-pipeline references, not decomposed factorial cells.*

**Manuscript:** Limitations §7.4 + Methods §5.2 factorial caveat; appendix Table “LifeManifold factorial cells” — see [`q1-v3-genetic-me-filter/ANALYSIS.md`](experiments/q1-v3-genetic-me-filter/ANALYSIS.md).

### 3.9 pyribs emitter configuration (disclosure) *[no mixed-emitter tier; F-RQ4 frozen]*

**Reviewer context:** modern pyribs pipelines sometimes use **heterogeneous** emitter pools (e.g. CMA-ME + `RandomDirectionEmitter` / `ImprovementEmitter`). Our B2 arms are **not** a single emitter.

| Knob | Locked (T0) |
|------|-------------|
| Scheduler | pyribs `Scheduler` with **`num_emitters = 5`** |
| Emitter type | **5× homogeneous `EvolutionStrategyEmitter`** (CMA-ES), batch **50** each → ask **250**, **130** asks → **32 500** evals |
| Algorithms | **CMA-ME** (`2imp` / filter / no_improvement) and **CMA-MAE** (result archive, `imp` / mu / basic) — separate frozen arms |
| vs LifeManifold `hints` | hints = **heterogeneous** ME (20 random + 20 genetic + 10 LLM); pyribs = **homogeneous gradient** pool on same 21-D adapter |

**What we do not claim:** “state-of-the-art pyribs emitter cocktail” or maximum pyribs QD ceiling. We claim **strong published CMA-ME/MAE references** on matched budget + warm-start.

**Optional post-arXiv (journal):** descriptive arm with mixed pyribs emitters (e.g. CMA + RandomDirection) — **does not amend F-RQ4** without §12; expected to **raise** the pyribs ceiling, not reverse F-RQ4 FAIL vs CMA-ME alone.

**Manuscript:** Methods disclosure + Limitations; cite [`Q1_V3_B2_PYRIBS_TASKS.md`](Q1_V3_B2_PYRIBS_TASKS.md) §Hyperparams (`num emitters = 5`).

### 3.10 QD-score reporting *[descriptive; post-hoc from archives]*

**Intent:** align tables/figures with QD literature (sum of elite fitness over filled cells) without amending frozen confirmatory families (still **coverage + mean_best_fitness**, conjunctive Holm).

| Item | Definition |
|------|------------|
| **QD-score** | \( \sum_{i \in \text{filled}} f_i \) = `filled_cells × mean_best_fitness` on **collapsed** archive (warm-start merge for grid runs) |
| **Primary (frozen)** | `coverage_pct`, `mean_best_fitness` — unchanged |
| **Descriptive** | `qd_score` column in `summary.csv`; optional Fig. / appendix ranking |
| **Progression** | `qd_score` field in `archive_trace.jsonl` (new runs); frozen matrix lacks traces — same as §3.7 |

**Implementation:** `worldspace/illuminators/archive_trace.py` (`qd_score_from_archive`); `scripts/aggregate_experiment_runs.py` merges grid baseline + run JSONL before summing; `scripts/plot_anytime_qd.py --metric qd_score`.

**Post-hoc (no re-run):** re-aggregate existing tiers, e.g. `uv run python scripts/aggregate_experiment_runs.py --root artifacts/experiments/q1-full`.

**Does not** replace Holm families or amend F-RQ0 / F-RQ1 / F-RQ4 without §12.

### 3.11 Archive heatmaps / qualitative BC coverage *[manuscript figures; no re-run]*

**Intent:** address reviewer expectation that scalar Δcov/Δfit hides **where** methods fill or miss niches in behaviour space — without new compute on the frozen matrix.

**Status:** **DONE 2026-07-22**

| Item | Plan |
|------|------|
| **Tooling** | `scripts/export_manuscript_figures.py --fig 7` (dashboard archive pivots → matplotlib) |
| **Pairs** | `hints` vs `cma_me` seeds **1 / 4 / 6** (smallest / largest / mid-large Δcov); optional `filter` vs `cma_me` seed **4** |
| **Deliverable** | Per-seed PDFs + `archive_heatmaps/*.png` + panel `fig07_archive_heatmaps_panel_seeds1_4_6.pdf`; main text Fig.~7 = seed 4 |
| **Not in scope (v3)** | Automated Coverage-Difference overlay |
| **Reproduce** | `.venv/bin/python scripts/export_manuscript_figures.py --fig 7 --seed 1 --seed 4 --seed 6 --panel` |

**Checklist:** §9 item «Archive heatmap figures».

### 3.12 B3 — Standard pyribs benchmarks (sphere + rastrigin) *[supplementary]*

**Status:** **DONE 2026-07-17** — runner/CLI/tiers wired; five 250-eval smoke combinations validated; full **50/50** matrix complete.

**Intent:** secondary **implementation / baseline validation** on Fontaine-style linear-projection illumination (pyribs tutorial lineage). Addresses “custom CA sandbox only” without claiming cross-domain LLM or surrogate generalization.

**What B3 validates:** pyribs CMA-ME/MAE runner and QD ranking (**CMA-ME ≳ MAP-Elites random**) on a literature-standard benchmark independent of the LifeManifold CA adapter.

**What B3 does *not* validate:** surrogate hints, LLM emitters, or CA-specific fitness/BC — **no LLM, no surrogate, no `WorldSpec` eval path**.

| Knob | Locked value |
|------|----------------|
| Library | **pyribs** `ribs==0.11.0` (B2 pin) |
| Benchmarks | **(1) sphere** (linear projection, Fontaine 2020 / pyribs tutorial) · **(2) rastrigin** (same measures, Rastrigin objective) |
| `solution_dim` | **20** (lightweight; not 100D tutorial default) |
| Archive | `GridArchive` **100×100**; measure ranges **±51.2** (`(D/2)×5.12`, D=20) |
| Eval budget | **32 500** evals/seed/arm (130 asks; CMA: 5×50; ME random: 1×250) |
| Seeds | **0–9** |
| Warm-start | **Empty archive** (no CA baseline JSONL) |
| Arms — **sphere** | `cma_me`, `cma_mae`, **`me_random`** (MAP-Elites + `GaussianEmitter`) |
| Arms — **rastrigin** | `cma_me`, `cma_mae` only |
| Surrogate / LLM | **Off** |
| Output | `artifacts/experiments/q1-v3-sphere/`, `artifacts/experiments/q1-v3-rastrigin/` |

**Task list / hyperparam lock:** [`Q1_V3_B3_SPHERE_TASKS.md`](Q1_V3_B3_SPHERE_TASKS.md) (T0 freeze **2026-07-14**).

```bash
# Smoke (seed 0):
PYRIBS_STANDARD_EVALUATIONS=250 ./scripts/run_experiment_batch.sh q1-v3-sphere 0 0
PYRIBS_STANDARD_EVALUATIONS=250 ./scripts/run_experiment_batch.sh q1-v3-rastrigin 0 0

# Full matrix (seeds 0–9):
./scripts/run_experiment_batch.sh q1-v3-sphere
./scripts/run_experiment_batch.sh q1-v3-rastrigin
```

**CLI:** `scripts/run_pyribs_standard.py` — `--benchmark {sphere,rastrigin}`, `--algo {cma_me,cma_mae,me_random}` (`me_random` rejected for rastrigin).

**Observed levels (mean ± SD, n=10):**

| Benchmark / arm | Coverage % | Mean objective | QD-score |
|-----------------|-----------:|---------------:|---------:|
| Sphere CMA-ME | **83.52 ± 2.96** | 71.69 ± 1.62 | **598,543 ± 19,033** |
| Sphere CMA-MAE | 34.98 ± 1.22 | **92.60 ± 0.30** | 323,890 ± 10,593 |
| Sphere ME random | 33.49 ± 0.63 | 91.88 ± 0.24 | 307,672 ± 5,111 |
| Rastrigin CMA-ME | **90.35 ± 1.84** | 56.72 ± 0.53 | **512,435 ± 9,978** |
| Rastrigin CMA-MAE | 32.17 ± 1.51 | **73.72 ± 0.60** | 237,070 ± 9,224 |

Sphere CMA-ME exceeds ME random by **+50.03 pp coverage** and **+290,872 QD-score** on 10/10 paired seeds (one-sided Wilcoxon p=0.00098; paired A₁₂=1.00), while mean objective is lower because CMA-ME fills harder niches. This is descriptive RQ-S evidence only; no Holm family is amended.

**Reporting (RQ-S, descriptive only):**

| Question | Comparison | Stats |
|----------|------------|-------|
| Does CMA-ME beat random MAP-Elites on sphere? | `cma_me` vs `me_random` | Within-domain Wilcoxon + A₁₂ (optional); **not** Holm vs CA |
| Are CMA levels literature-plausible? | levels vs pyribs tutorial order-of-magnitude | Qualitative |
| CA vs sphere ranking | cma_me (CA B2) vs cma_me (sphere) | **Levels table only** — not paired cross-domain |

**Manuscript:** supplementary § / appendix — *“Primary LLM+surrogate evaluation remains on stochastic CA; standard sphere/rastrigin runs validate pyribs baseline implementation only.”*

**Does not:** amend F-RQ0 / F-RQ4 / RQ1 / F-RQ1g; replace B2 on CA; require LLM on second domain.

---

## 4. Statistics lock — FROZEN 2026-07-12

> **Rule:** §§1–4 frozen **2026-07-12**. Any change requires §12 amendment date + dual-report of old vs new text. Starting the first v3 seed under an undeclared edit invalidates confirmatory language for that family.

### 4.1 Confirmatory families

| Family | Tests | m (Holm) | Primary archive |
|--------|-------|----------|-----------------|
| **F-RQ0** | vanilla vs hints: Δcov and Δfit, conjunctive (both one-sided greater) | 2 | grid |
| **F-RQ4** | hints vs CMA-ME and hints vs CMA-MAE: for each arm, Δcov and Δfit (hints − arm), conjunctive greater; two arms → **m=4** | 4 | grid |
| **F-RQ1g** | per additional LLM: stub vs hints Δcov+Δfit conjunctive | 2 × n_llm | grid |
| **F-RQ3** (historical; production filter) | eval↓; cov NI; fit NI | 3 | grid — **BLOCKED** (D1@0.45 fail) |
| **F-RQ3-gray** *(2026-07-28)* | `filter_gray_zone` vs `hints`: eval↓; cov NI; fit NI | 3 | grid — **confirmatory DONE** ($n{=}10$; tier `q1-v3-h3-gray-zone`); pilot exploratory DONE |

Holm step-down within each family separately (do not pool B1+B2+G1 into one mega-family unless explicitly amended).

**F-RQ4 metric set (locked):** same primary metrics as v2 — `coverage_pct`, `mean_best_fitness`. Eval count is descriptive for pyribs (budget-matched by design).

### 4.2 D1 gate — historical (production filter @ combat `min_fit=0.45`)

v2 established that **combat filter = `min_predicted_fitness=0.45`** (raised from 0.10 after shadow; logged skips agree=1.0). D1 must use the **same** threshold as combat — not the historical 0.10 draft in early §3.5.

| Item | v3 lock |
|------|---------|
| Combat `min_predicted_fitness` | **0.45** (inherit v2) |
| D1 confirmatory metric (historical) | live-proposal `divergent_skip` @ **min_fit=0.45**, gates 0.5 vs 0.95 |
| Pass rule (historical) | `div ≤ 0.05` → was required for F-RQ3 on production filter |
| Primary artifact | `compose_gate_live_0p5_vs_0p95.json` — DONE, 10 seeds / 325k proposals, div=**0.528**, **FAIL** |
| Sensitivity only | `@min_fit=0.10` → div=**0.746** (worse; not combat; do not use for gate) |

**Status (2026-07-28):** D1 remains a **documented diagnostic** on locked `q1-full/filter` logs. It **does not** gate the amended confirmatory path §4.2.1. Hold-out compose gate alignment (§12 2026-07-28) fixes validity reporting only.

**Do not** re-lock combat or D1 to 0.10: that mismatches the −33.5% eval filter arm and **inflates** divergent_skip.

**Shadow (production filter):** keep skip in 25–45% under **0.45** if filter is re-run; no forced return to 0.10.

### 4.2.1 F-RQ3-gray — amended H3 path *(2026-07-28; journal scope 2026-07-29)*

**Submission scope (2026-07-31):** This revision reports the diagnosed compose-gate mechanism (dual definition; gray-zone $74.6\%$; in-band vs out-of-band $d$), hold-out alignment to gate **0.95**, honest NMAE degradation, an exploratory **`filter_gray_zone`** pilot ($n{=}10$; `q1-v3-h3-gray-zone-pilot`), and a **confirmatory duplicate** tier ($n{=}10$; `q1-v3-h3-gray-zone`) with F-RQ3-gray Holm/NI read (artifacts: `h3_gray_zone_confirmatory_holm.json`, `H3_GRAY_HOLM.md`). **Locked production H3 remains blocked** (D1@0.45 fail on historical `q1-full/filter`). Passing F-RQ3-gray does **not** rehabilitate production `filter` @ 33.5% skip.

**Confirmatory results (2026-07-31, tier `q1-v3-h3-gray-zone`):** skip **11.90 ± 0.20%** (target 8–18%); mean $\Delta$cov vs frozen hints **−0.60 pp** (2/10 wins); F-RQ3-gray Holm $m{=}3$ **pass** (eval ↓, cov NI, fit NI) — **not** a coverage win and **not** rehabilitation of production `filter`.

**Pilot results (2026-07-29, exploratory):** skip **11.69 ± 0.16%**; mean $\Delta$cov **−0.52 pp** (3/10 wins); exploratory F-RQ3-gray Holm **pass** — directional replication of confirmatory tier.

**Dual-report (replaces D1-unlock path for new runs):**

| | **Before (§4.2 through 2026-07-27)** | **After (2026-07-28)** |
|--|--------------------------------------|-------------------------|
| Confirmatory H3 | Blocked until D1@0.45 ≤ 0.05 on production filter | **F-RQ3-gray** family with `force_eval_extinction_gray_zone: true` |
| Filter policy | `threshold_gate` @ τ=0.45, compose gate 0.95 | Same τ + compose gate + **never skip** when $p_{\mathrm{ext}}\in[0.5,0.95)$ |
| Pre-flight skip band | 25–45% (production filter) | **8–18%** (offline replay ~11.7%; shadow seed 0 before n=10) |
| Control | `hints` (`q1-full`) | unchanged — reuse frozen arm |
| Treatment tier | `q1-full/filter` (exploratory only) | **`q1-v3-h3-gray-zone`** |

| Item | Lock |
|------|------|
| Scheduler YAML | `map_elites_scheduler_nightly_llm_filter_gray_zone.yaml` |
| Implementation | `threshold_gate_gray_zone_v1`; reason `extinction_gray_zone_force_eval` |
| Endpoints | eval count ↓ (Wilcoxon less); cov NI (Δ > −3 pp); fit NI (Δ_rel > −5%) — v2 amended margins |
| Holm | $m=3$ within **F-RQ3-gray** only |
| Offline justification | `compose_gate_fix_candidates.json`: gray-zone D1=0.708; force-eval policy D1=0 |
| Task list | [`Q1_H3_GRAY_ZONE_CONFIRMATORY.md`](Q1_H3_GRAY_ZONE_CONFIRMATORY.md) |

**Explicit:** passing F-RQ3-gray does not rehabilitate the historical production `filter` arm at 33.5% skip. Soft-extinction retarget remains exploratory.
### 4.3 Non-inferiority / TOST (RQ3 if reopened)

Prefer v2 **amended one-sided NI** as paper claim (Δcov > −3 pp, Δfit_rel > −5%); keep symmetric TOST as appendix transparency. Declare before runs.

### 4.4 Multi-LLM reporting

- Primary table: per-model Δcov / Δfit + Holm within F-RQ1g.
- Do not average across LLMs for confirmatory pass/fail.
- Pin API window dates per model (floating aliases forbidden for new G1 runs).

### 4.5 Descriptive effect sizes (reporting only) *[2026-07-14]*

**Intent:** QD reporting convention — non-parametric effect size alongside Wilcoxon p-values; **does not** amend confirmatory gates (still Holm on Wilcoxon only).

| Item | Lock |
|------|------|
| **Primary descriptive ES** | Paired **Vargha–Delaney A₁₂** on matched seeds: P(treatment > control) + 0.5·P(tie) per metric |
| **Direction** | Same as Wilcoxon alternative (`greater` for hints−stub/cma; `less` for eval filter−hints) |
| **Interpretation** | A₁₂ = 0.5 null; ~0.56 small, ~0.64 medium, ~0.71 large (Vargha & Delaney 2000) |
| **Also report** | Median paired Δ + bootstrap 95% CI (existing) |
| **Supplementary** | Cohen’s dz (paired) — retained for `local_ok` bar (dz≥0.5); appendix only in manuscript tables |
| **Implementation** | `scripts/analyze_q1_statistics.py` (`vargha_delaney_a12_paired`); all families (RQ1, F-RQ4, CVT descriptive) |

**Does not** change Holm α, family definitions, or pass/fail rules in §4.1.

---

## 5. Experiment tiers (names — implement in `run_experiment_batch.sh`)

| Tier | Seeds | Conditions | Notes |
|------|-------|------------|-------|
| `q1-v3-vanilla` | 0–9 | vanilla | B1 / RQ0 random-only ME |
| `q1-v3-pyribs` | 0–9 | cma_me, cma_mae | B2 (CA adapter) |
| `q1-v3-sphere` | 0–9 | cma_me, cma_mae, me_random | B3 sphere — **DONE 30/30** |
| `q1-v3-rastrigin` | 0–9 | cma_me, cma_mae | B3 rastrigin — **DONE 20/20** |
| `q1-v3-llm-deepseek-v4-pro` | 0–4 default / 0–9 full | stub, hints | G1 DeepSeek V4 Pro |
| `q1-v3-llm-gpt-4o-mini` | 0–4 default / 0–9 full | stub, hints | G1 OpenAI gpt-4o-mini |
| `q1-v3-llm-<slug>` | 0–9 (or 0–4) | stub, hints | G1 other models |
| `q1-v3-h3-gray-zone` | 0–9 | `filter_gray_zone` | **F-RQ3-gray** confirmatory **DONE** ($n{=}10$); Holm/NI in `H3_GRAY_HOLM.md` |
| `q1-v3-h3-gray-zone-pilot` | 0–9 | `filter_gray_zone` | exploratory pilot **DONE** ($n{=}10$); Holm/NI in `H3_GRAY_HOLM.md` |
| `q1-v3-filter` | 0–9 | stub, hints, filter | historical production filter; exploratory only |
| `cvt-shadow-v3` | 0 | shadow pair | if CVT filter revisited with new threshold |

Reuse v2 `q1-full` / `q1-cvt` artifacts for paired contrasts where seeds match; keep filter arms at **min_fit=0.45** (do not mix a 0.10 policy into confirmatory tables).

---

## 6. Cost sketch (indicative)

| Block | Runs (order) | LLM calls | Dominant cost |
|-------|--------------|-----------|---------------|
| B1 vanilla | 10 | 0 | sim ~same as one stub arm |
| B2 pyribs | 20 | 0 | sim + CMA overhead |
| B3 sphere + rastrigin | 50 | 0 | cheap analytic eval (~1–3 h wall) |
| G1 (2 LLMs × 10 seeds × stub+hints) | 40 | ~260 000 | API |
| G1 minimal (2 LLMs × 5 seeds × 2) | 20 | ~130 000 | API |
| M1 / R1 | 0–few | 0 | author time |

v2 already spent ~390 000 LLM calls (grid+CVT). v3 LLM marginal ≈ **G1 only** unless filter re-run.

---

## 7. MC-dropout vs GP+UCB (M1)

*Filled 2026-07-22 (Phases 0–4). Details: `artifacts/surrogate/M1_ANALYSIS.md`; manuscript: `draft_v0.tex` §acquisition.*

1. **Scale/cost:** nightly buffer ~35k × 24-dim → MLP fitness batch @32.5k **~1 s** vs GP **~11 s**; full MC-dropout u every slot would be ~tens of minutes.
2. **Q1 acquisition:** `max_uncertainty_to_skip=1.0` → σ not load-bearing for `threshold_gate`; GP+UCB would change the **scientific claim**, not only the regressor.
3. **Hold-out:** MLP R² **0.76** vs GP **0.22**; threshold_gate false-skip **0.7%** (MLP) vs ~**3%** (GP / GP+UCB).
4. **UCB offline:** β∈{0.15,0.5,1.0} on GP stays ~99% skip; does not beat threshold+MLP on false-skip.
5. **UCB live (logged MLP):** filter skip **33.5%**; UCB softens to **28.5/25.7/21.2%** (β=0.15/0.5/1.0); no eval→skip flips. Not a GP online claim.
6. **Manuscript:** justification paragraph in `draft_v0.tex` (§acquisition + Limitations).

---

## 8. Related work (R1 — SAIL / DSA-ME)

*Filled 2026-07-22. Manuscript: `draft_v0.tex` §2.2.*

Surrogate-assisted QD includes SAIL (Gaussian-process surrogates with acquisition over the behaviour–fitness landscape; typically UCB-style eval selection) and DSA-ME / deep-surrogate MAP-Elites (online neural predictors with an outer validation loop). Our setting differs in four respects: (i) the simulator is a stochastic continuous CA with genome-parameterized physics and seeded deterministic eval; (ii) variation includes LLM emitters with optional surrogate **hints** in-prompt (Role 1) as well as an optional **threshold filter** on composed MLP predictions (Role 2)—not a GP-UCB illuminator (M1 offline ablation prefers MLP over GP and rejects GP+UCB as a drop-in); (iii) archive insertion always uses simulated metrics; (iv) we disclose an explicit compose-gate mismatch (D1) between offline and online extinction thresholds. We therefore treat SAIL/DSA-ME as related surrogate-QD lineages rather than drop-in baselines; pyribs CMA-ME/CMA-MAE (§3.2) are the quantitative QD baselines for v3.

---

## 9. Reproducibility checklist (v3)

- [x] Freeze §§1–4; record freeze date: **2026-07-12**
- [x] **External protocol timestamp:** none for this submit (**2026-07-22**) — keep *internally frozen*; OSF/Zenodo deferred
- [x] Pin pyribs version + CMA-ME/MAE hyperparameters — **`ribs==0.11.0`**; full lock in [`Q1_V3_B2_PYRIBS_TASKS.md`](Q1_V3_B2_PYRIBS_TASKS.md) §Hyperparams + **§3.2 inline table**; frozen B2 JSONs predate the full `pyribs_hyperparams` block (no re-run)
- [x] Add `vanilla` scheduler YAML + batch tier (`map_elites_scheduler_nightly_vanilla.yaml`, `q1-v3-vanilla`)
- [x] RQ0 / B1 vanilla matrix complete (seeds 0–9) + F-RQ0 PASS — see `q1-v3-vanilla/ANALYSIS.md`
- [x] Add `run_pyribs_baseline.py` + golden parity test (fitness/BC) — T1 adapter + T2 runner (`tests/test_pyribs_adapter.py`, `tests/test_pyribs_baseline.py`)
- [x] Wire batch tier `q1-v3-pyribs` → `{cma_me,cma_mae}/seed_*` (`run_experiment_batch.sh`; aggregate recognizes `cma_me`/`cma_mae`)
- [x] B2 matrix seeds **0–9** × `cma_me`/`cma_mae` @ 32 500 + descriptive `q1-v3-pyribs/ANALYSIS.md`
- [x] Holm family **F-RQ4** coded + reported (`analyze_q1_statistics.py --family frq4`) — **FAIL** (CMA-ME dominates hints; only `RQ4_mae_fit` Holm-rejects)
- [x] Choose ≥2 G1 model IDs (**dated**); document `api_base` — DeepSeek: `deepseek-v4-pro` @ `api.deepseek.com` (wired); OpenAI: `gpt-4o-mini` @ `api.openai.com` (wired; smoke response `gpt-4o-mini-2024-07-18`)
- [x] Wire DeepSeek G1 tier `q1-v3-llm-deepseek-v4-pro` + `llm_world_generator_deepseek.yaml` (`thinking: {type: disabled}`)
- [x] Wire OpenAI G1 tier `q1-v3-llm-gpt-4o-mini` + `llm_world_generator_openai.yaml`
- [x] G1-minimal DeepSeek run + pin (`deepseek-v4-pro`, `api.deepseek.com/v1/chat/completions`, hash `fd5f0d34f97f9df7`, window 2026-07-12 03:11–18:54) — see `q1-v3-llm/deepseek-v4-pro/ANALYSIS.md`
- [x] G1-minimal OpenAI gpt-4o-mini complete (seeds 0–4, stub+hints, 10/10; hash `b678837d32da4f49`, window 2026-07-13→14) — see `q1-v3-llm/gpt-4o-mini/ANALYSIS.md`
- [x] **G1 arXiv freeze (2026-07-14):** no further G1 compute before preprint; seeds 0–9 + F-RQ1g Holm per model → journal revision
- [x] **Target-selection disclosure (2026-07-14):** §3.6 documents stub `min_fitness_frontier` vs hints `uniform_frontier`; manuscript §5.2/§7.4
- [x] Wire post-arXiv `stub_uniform` sensitivity: `map_elites_scheduler_nightly_llm_stub_uniform.yaml` + tier `q1-stub-uniform-sensitivity`
- [x] **stub_uniform matched sensitivity:** seeds **0–9 DONE** (2026-07-19) → mean Δcov(hints−stub_uniform) **+0.26 pp**, Δfit **+0.0033**, Wilcoxon p=0.41; descriptively flat (not confirmatory NULL)
- [x] **RQ1b hints_rich pilot (2026-07-15):** seed 0 NULL; scheduler + tier + [`q1-hints-rich-pilot/ANALYSIS.md`](experiments/q1-hints-rich-pilot/ANALYSIS.md); §3.6b
- [x] **RQ1d hints_parent pilot (2026-07-15):** seed 0 NULL; pre-flight GO (median \|Δfit\| 0.127) but QD unchanged; [`q1-hints-parent-pilot/ANALYSIS.md`](experiments/q1-hints-parent-pilot/ANALYSIS.md); §3.6c
- [x] **RQ1e hints_direction pilot (2026-07-15):** seed 0 NULL; pre-flight GO (median max \|∂fit\| 0.783) but QD unchanged; [`q1-hints-direction-pilot/ANALYSIS.md`](experiments/q1-hints-direction-pilot/ANALYSIS.md); §3.6d
- [x] **RQ1f weak-model pilot (§3.6e):** seeds 0–2 NULL (omni-7b); Δcov −0.08 pp, interaction +0.07 pp; [`q1-v3-llm-weak-pilot/ANALYSIS.md`](experiments/q1-v3-llm-weak-pilot/ANALYSIS.md)
- [x] B1b genetic ME wired + seeds 0–9 DONE — `map_elites_scheduler_nightly_genetic_me.yaml`, tier `q1-v3-genetic-me`
- [x] **Factorial `genetic_me_filter` (§3.8):** seeds 0–9 DONE; historical (target-selection-confounded) Δcov vs `genetic_me` **+12.28 pp**; [`q1-v3-genetic-me-filter/ANALYSIS.md`](experiments/q1-v3-genetic-me-filter/ANALYSIS.md)
- [x] **B1b `genetic_me`:** seeds 0–9 DONE — mean cov **45.93%**; [`q1-v3-genetic-me/ANALYSIS.md`](experiments/q1-v3-genetic-me/ANALYSIS.md)
- [x] **Matched factorial control `genetic_me_uniform` (§3.8):** seeds 0–9 DONE; filter +3.65 pp @ 20k eval, −0.83 pp at 650 iterations; [`ANALYSIS.md`](experiments/q1-v3-genetic-me-uniform/ANALYSIS.md)
- [x] **pyribs emitter disclosure (§3.9):** 5× homogeneous ES documented; mixed RDA/Improvement pool out of scope / post-arXiv optional
- [x] **QD-score (§3.10):** collapse-aware `qd_score` in aggregate + trace; re-aggregate frozen runs (no confirmatory amendment)
- [x] **Vargha–Delaney A₁₂ (§4.5):** paired effect size in `analyze_q1_statistics.py`; reporting only (Holm unchanged)
- [x] **Archive heatmap figures (§3.11):** seeds 1/4/6 hints vs CMA-ME + panel; optional filter seed 4; under `artifacts/manuscript/figures/archive_heatmaps/`
- [x] **B3 standard benchmarks (§3.12):** runner + tiers wired; smoke validated; **50/50 runs** + both ANALYSIS files complete (supplementary sanity PASS)
- [x] Wire anytime `archive_trace.jsonl` logging (MAP-Elites + pyribs) + `scripts/plot_anytime_qd.py` — **do not re-run** frozen matrix before arXiv
- [ ] Shadow-calibrate if filter re-run (target skip 25–45% @ **min_fit=0.45**)
- [x] **D1 live replay @0.45 DONE:** `compose_gate_live_0p5_vs_0p95.json`; divergence **0.528** > 0.05 → confirmatory RQ3 **blocked** (0.10 sensitivity-only)
- [x] Holm families: **F-RQ4 coded** (`--family frq4`); **F-RQ1g coded** (`--family frq1g`); F-RQ0 reported in analysis but not centralized
- [x] **M1 GP+UCB rationale DONE** — Phases 0–4; `gp_ucb_ablation.json` + paragraph in `draft_v0.tex` §acquisition
- [x] **R1 SAIL/DSA-ME DONE** — expanded `draft_v0.tex` §2.2 + protocol §8
- [x] No floating `qwen-turbo` for new G1 runs — DeepSeek uses explicit `deepseek-v4-pro`; OpenAI uses `gpt-4o-mini` (response snapshot `gpt-4o-mini-2024-07-18`)
- [ ] Git SHA logged in `nightly_run_summary.json` (v2 open item)

---

## 10. Relation to v2

| Topic | v2 | v3 |
|-------|----|----|
| Primary matrix | Grid stub/hints/filter + CVT-s | **Add** vanilla, pyribs, multi-LLM |
| RQ3 status | Exploratory (D1@0.45) | Reopen only if D1@**0.45** passes |
| Filter threshold | Combat **0.45** | **Inherit 0.45** (0.10 superseded) |
| LLM generalizability | Single floating `qwen-turbo` window | ≥2 additional pinned models |
| QD baselines | Internal stub only | Vanilla ME + CMA-ME/MAE (CA) + sphere/rastrigin (B3) |
| Related work depth | Light | SAIL/DSA-ME paragraph + GP+UCB rationale |

---

## 11. Quick decision tree

```text
Need camera-ready QD claim?     → finish B1 + B2 first (CPU)
Need “LLM helps” general claim? → G1 (≥2 models, dated IDs)
Need stronger novelty vs SAIL?  → M1 + R1 DONE in draft_v0.tex (§acquisition, §2.2)
Want confirmatory RQ3 in v3?    → fix compose mismatch first; D1@0.45 already 0.528 (0.10 worse)
Budget tight on API?            → G1 with 5 seeds; slip filter re-run
Need “not custom sandbox only”? → B3 sphere + rastrigin (CPU, no LLM)
```

---

## 12. Amendment log

| Date | Change |
|------|--------|
| 2026-07-12 | Initial v3 stub: B1/B2/G1/M1/R1 + stats lock |
| 2026-07-12 | Fix: D1/combat lock **0.45** (not 0.10); 0.10 sensitivity-only |
| **2026-07-12** | **Freeze §§1–4.** Status → FROZEN. F-RQ4 locked to m=4 (cov+fit × CMA-ME/MAE). Further edits to §§1–4 need amendment + dual-report. |
| 2026-07-12 | G1 DeepSeek ready: `llm_world_generator_deepseek.yaml` + tier `q1-v3-llm-deepseek-v4-pro`; thinking toggle via `thinking: {type: disabled}` |
| 2026-07-12 | Smoke: live DeepSeek ping **PASS** (`thinking` disabled; `enable_thinking:false` alone insufficient) |
| 2026-07-12 | G1-minimal DeepSeek **complete** (seeds 0–4): pin `deepseek-v4-pro` @ `api.deepseek.com/v1/chat/completions`, hash `fd5f0d34f97f9df7`, window 2026-07-12 03:11–18:54; Δcov +16.46 pp / Δfit +0.104 (5/5). Analysis: `q1-v3-llm/deepseek-v4-pro/ANALYSIS.md` |
| 2026-07-12 | G1 OpenAI ready: `llm_world_generator_openai.yaml` + tier `q1-v3-llm-gpt-4o-mini`; smoke PASS (`gpt-4o-mini` → response `gpt-4o-mini-2024-07-18`); hash `b678837d32da4f49` |
| 2026-07-12 | B1 / RQ0 ready: `map_elites_scheduler_nightly_vanilla.yaml` (50× random, llm off) + tier `q1-v3-vanilla` (seeds 0–9) |
| 2026-07-12 | RQ0 complete: vanilla n=10; F-RQ0 PASS (hints−vanilla Δcov +18.05 pp / Δfit +0.166, 10/10, Holm m=2). Analysis: `q1-v3-vanilla/ANALYSIS.md` |
| 2026-07-12 | **B2 T0 freeze:** §3.2 dual-report — **was:** “Report hints vs each pyribs arm … (two-sided or TOST non-inferiority — **choose before runs**, §4).” **now:** F-RQ4 per §4.1 — one-sided **greater** Δcov+Δfit (`hints` − arm), conjunctive, Holm **m=4**. Also lock: 21-D genetic genome + bit-rounding disclosure; warm-start grid baseline; hyperparams in `Q1_V3_B2_PYRIBS_TASKS.md`. |
| 2026-07-12 | **B2 T3.1–T3.3:** wire batch tier `q1-v3-pyribs` in `run_experiment_batch.sh` → `{cma_me,cma_mae}/seed_*`; `aggregate_experiment_runs.py` recognizes `cma_me`/`cma_mae` (and `vanilla`); smoke seed 0 with `PYRIBS_EVALUATIONS=250`. Full 0–9 matrix remains T3.4. |
| 2026-07-13 | **B2 T3.4–T3.5:** full matrix seeds **0–9** × `cma_me`/`cma_mae` @ 32 500 (20/20); descriptive analysis `q1-v3-pyribs/ANALYSIS.md` (levels + Δ vs v2 hints). F-RQ4 confirmatory remains T4. |
| 2026-07-13 | **B2 T4 / F-RQ4 FAIL:** `analyze_q1_statistics.py --family frq4`; Holm m=4 — only `RQ4_mae_fit` rejects; CMA-ME Δcov mean **−12.96** pp (hints behind); MAE Δfit **+0.050** (hints ahead on fit only). Artifacts: `q1-v3-pyribs/frq4_statistics.json`, ANALYSIS §F-RQ4. Pre-registered claim (hints > both CMA arms) not supported. |
| 2026-07-14 | **G1-minimal gpt-4o-mini complete** (seeds 0–4, stub+hints, 10/10): Δcov **+11.09 pp** / Δfit **+0.070** (5/5); direction replicates RQ1, absolute hints **~55%** vs **~60%** qwen/DeepSeek. Analysis: `q1-v3-llm/gpt-4o-mini/ANALYSIS.md`. |
| 2026-07-14 | **G1 arXiv freeze:** both additional LLMs reported at n=5 **exploratory**; no further G1 runs before preprint; F-RQ1g Holm per model + seeds 0–9 deferred to journal revision. Manuscript snippets in `artifacts/manuscript/SWARM_EC_DRAFT_OUTLINE.md` §6.4, §7.4. |
| 2026-07-14 | **Target-selection confound:** historical `stub` uses default `min_fitness_frontier`; `hints`/`filter` use `uniform_frontier`. Matched `stub_uniform` tier added; **superseded by completed n=5 sensitivity on 2026-07-17**. |
| 2026-07-14 | **B1b genetic ME:** add `genetic_me` arm (20R+30G, no LLM) — `map_elites_scheduler_nightly_genetic_me.yaml` + tier `q1-v3-genetic-me`; descriptive baseline for canonical archive-variation ME; does not amend F-RQ0 (random floor unchanged). |
| 2026-07-14 | **Anytime trace wired:** `archive_trace.jsonl` in MAP-Elites loop + pyribs runner when timing enabled; plot script `plot_anytime_qd.py`. Frozen matrix lacks traces; selective CPU re-run deferred (post-arXiv / journal). |
| 2026-07-14 | **Factorial ablation (−LLM/+filter):** `genetic_me_filter` arm wired; descriptive only, not in frozen Holm families. **Superseded by completed matched n=10 acquisition analysis on 2026-07-17**. |
| 2026-07-14 | **pyribs emitter disclosure (§3.9):** B2 uses **5× `EvolutionStrategyEmitter`** (not single-emitter); no heterogeneous pyribs pool (RDA/Improvement mix) in frozen F-RQ4; optional journal sensitivity only. |
| 2026-07-14 | **QD-score (§3.10):** descriptive `qd_score` (= sum elite fitness on collapsed archive) in aggregate + `archive_trace`; confirmatory endpoints remain coverage+mean conjunctive; post-hoc re-aggregate frozen tiers. |
| 2026-07-14 | **Archive heatmaps (§3.11):** manuscript reminder — qualitative BC coverage via existing Streamlit Archive Explorer (paired-seed screenshots hints vs CMA-ME); no CD diff script in v3; author-time before arXiv. |
| 2026-07-14 | **Vargha–Delaney A₁₂ (§4.5):** descriptive paired effect size in `analyze_q1_statistics.py`; Wilcoxon/Holm confirmatory gates unchanged; dz retained for `local_ok` / appendix. |
| 2026-07-14 | **B2 hyperparam disclosure:** §3.2 inline σ₀/λ/restart table + task-doc lock. Frozen B2 JSONs predate the full `pyribs_hyperparams` block; no re-run. |
| 2026-07-14 | **Factorial vs CMA FAQ (§3.8):** reviewer disclosure — LifeManifold 2×2 cells vs cross-pipeline CMA; arXiv = text only; `genetic_me_filter` journal priority. |
| 2026-07-14 | **B3 standard benchmarks (§3.12):** sphere + rastrigin pyribs arms (CMA-ME/MAE; **me_random** on sphere only); D=20, 100×100 archive, 32 500 evals, empty warm-start; **no LLM/surrogate**; descriptive RQ-S only. Task list: [`Q1_V3_B3_SPHERE_TASKS.md`](Q1_V3_B3_SPHERE_TASKS.md). Tiers: `q1-v3-sphere`, `q1-v3-rastrigin`. Does not amend frozen confirmatory families. |
| 2026-07-15 | **RQ1b hints_rich pilot (§3.6b):** `map_elites_scheduler_nightly_llm_hints_rich.yaml` + tier `q1-hints-rich-pilot`; seed 0 only — cov **60.12%**, fit **0.4958**, user hash `39effe9f` (components template active). NULL vs +2 pp threshold vs hints/stub_uniform. **No** seeds 1–4. Manuscript: soften causal hint claims; emphasize production stack vs vanilla/CMA. |
| 2026-07-15 | **RQ1d hints_parent pilot (§3.6c):** parent observed metrics in hint header; pre-flight GO (median \|parent−surrogate fit\| **0.127**); seed 0 — cov **59.92%**, fit **0.4945**, hash `6197d5e2`. NULL vs hints/stub_uniform/hints_rich. **Dual-report:** pre-amendment Path 4D hoped salient observed metrics would activate LLM channel; **post-amendment:** discriminative numbers still ≈ NULL QD — **hint channel closed** for scalar, component-rich, and parent-header formats. No seeds 1–4. |
| 2026-07-15 | **RQ1e hints_direction pilot (§3.6d):** FD direction-of-improvement hints; pre-flight GO (median max \|∂fit\| **0.783**); seed 0 — cov **59.04%**, fit **0.5040**, hash `1ebbb172`. NULL vs hints (Δcov −0.68 pp, Δfit +0.006). **Post-amendment:** strong surrogate gradients do not translate to QD via LLM — **hint channel closed** for all tested prompt formats; stop Path 4/5 prompt engineering. No seeds 1–4. |
| 2026-07-15 | **RQ1f weak-model interaction (§3.6e):** infrastructure wired; **superseded by completed pilot on 2026-07-16**. |
| 2026-07-16 | **RQ1f weak-model pilot complete (§3.6e):** `qwen2.5-omni-7b` (7B instruct **403**); pre-flight GO (5/5); seeds 0–2 — hints **56.97%** vs stub_uniform **57.05%** (Δcov **−0.08 pp**); interaction vs strong **+0.07 pp**. **NULL** — model-capability moderation rejected; **hint channel closed** for capable + weak models. Analysis: `q1-v3-llm-weak-pilot/ANALYSIS.md`. Manuscript §6.9/§7.2/§7.4 updated. |
| 2026-07-17 | **Matched acquisition control complete (§3.8):** `genetic_me_uniform` vs `genetic_me_filter`, n=10. Filter saves **33.5%** sims for **−0.83 pp** final coverage at fixed 650 iterations; at equal eval budget it leads **+3.65 pp @ 20k**, reaches 55% coverage with **29% fewer evals**, but is **3.2× slower wall-clock** due to MLP overhead. Acquisition channel PASS for sample efficiency. Historical +12.28 pp vs `genetic_me` remains target-selection-confounded. |
| 2026-07-17 | **stub_uniform sensitivity (§3.6):** seeds **0–4**; mean Δcov(hints−stub_uniform) **+0.57 pp**, Δfit **+0.0036**. Matched scalar hint channel descriptively flat; frozen F-RQ1 remains a historical bundled contrast. |
| 2026-07-19 | **stub_uniform sensitivity extended (§3.6):** seeds **0–9 DONE**; mean Δcov(hints−stub_uniform) **+0.26 pp**, Δfit **+0.0033**, Wilcoxon p=0.41. Target-selection confound confirmed at F-RQ1 power (stub_uniform−stub **+14.83 pp**). Analysis: `q1-stub-uniform-sensitivity/ANALYSIS.md`. |
| 2026-07-19 | **Anytime ladder queued:** tier `q1-anytime-ladder` (vanilla + hints + cma_me, seeds 0–4 default) → `artifacts/experiments/q1-anytime-ladder/`; does not overwrite frozen `q1-full`. |
| 2026-07-20 | **Anytime ladder complete (§3.7):** seeds 0–4 DONE; 15/15 runs; terminal parity vs frozen (vanilla/cma_me exact; hints +0.19 pp mean). Median coverage @ 10k/32.5k: vanilla 40.0/42.2%, hints 47.2/60.0%, cma_me 60.9/71.4%. Fig. 8 + `q1-anytime-ladder/ANALYSIS.md`. |
| 2026-07-20 | **CMA encoding ablation wired (§3.7b):** `decode_mode` {rint,threshold,bernoulli} + tier `q1-cma-encoding-ablation`. |
| 2026-07-21 | **G1 gpt-4o-mini extended (§3.3):** seeds **5–9 DONE**; matrix **n=10** complete. mean Δcov **+9.66 pp**, Δfit **+0.018**; Wilcoxon p_cov=0.001, p_fit=0.003; **F-RQ1g PASS** (Holm m=2). |
| 2026-07-22 | **G1 DeepSeek extended (§3.3):** seeds **5–9 DONE**; matrix **n=10** complete. mean Δcov **+16.95 pp**, Δfit **+0.052**; Wilcoxon p≈0.001 both; **F-RQ1g PASS** (Holm m=2). G1 critical gate **DONE** (both providers). Analysis: `deepseek-v4-pro/ANALYSIS.md`, `frq1g_statistics.json`. |
| 2026-07-22 | **M1 Phase 0–1:** scope lock `artifacts/surrogate/M1_PLAN.md`; offline MLP vs GP hold-out via `scripts/compare_surrogate_regressors.py` → `gp_ucb_ablation.json` + `M1_ANALYSIS.md`. MLP R² **0.761** / MAE **0.0069** vs GP (n=5k) R² **0.223** / MAE **0.045**; MLP fitness batch @32.5k **~12×** faster. Phase 2 (UCB replay) still open. |
| 2026-07-22 | **M1 Phase 2:** implement `ucb_promote` (`μ+βσ`); `scripts/compare_acquisition_policies_m1.py`; offline replay on same hold-out. threshold_gate+MLP skip **0.967** / false-skip **0.007**; GP+UCB β∈{0.15,0.5,1.0} skip ~**0.99** / false-skip ~**0.03**; none in 25–45% shadow band. Agreement vs UCB(GP,0.15)=**0.972**. Paragraph drafted in `M1_ANALYSIS.md`. |
| 2026-07-22 | **F-RQ1g coded:** `analyze_q1_statistics.py --family frq1g` → per-provider + combined `q1-v3-llm/frq1g_statistics.json`; both G1 providers **PASS** @ n=10. |
| 2026-07-22 | **M1 Phase 3:** `scripts/compare_acquisition_live_m1.py` on `q1-full/filter` (n=325k). threshold_gate agree_logged=**1.0**; UCB on logged MLP softens skip **0.335→0.285/0.257/0.212**; no GP re-predict (no features). |
| 2026-07-22 | **M1 Phase 4 / DONE:** paste GP+UCB justification into `draft_v0.tex` (§acquisition + Limitations); protocol M1 closed. |
| 2026-07-22 | **R1 DONE:** expand SAIL/DSA-ME §2.2 in `draft_v0.tex` (4-way positioning + M1/D1 cross-links); fill protocol §8. |
| 2026-07-22 | **§3.11 heatmaps DONE:** seeds 1/4/6 hints vs CMA-ME + panel; filter seed 4 optional; `export_manuscript_figures.py --fig 7`. |
| 2026-07-22 | **Data/code availability + freeze policy:** expanded `draft_v0.tex` §Data (tier/stats/M1 commands); **no OSF/Zenodo** for this submit — keep **internally frozen** (v3 2026-07-12; v4 2026-07-17). |
| 2026-07-28 | **M1 hold-out recomputed @ gate 0.95:** MLP R²=**0.942** / NMAE=**0.112** vs GP R²=**0.891** / NMAE=**0.222** (~37× faster). Legacy @0.5 (0.761 vs 0.223) retired as model-quality contrast — target-rescaling artifact. Artifact `gp_ucb_ablation.json` updated. |
| 2026-07-28 | **F-RQ3-gray path (§4.2.1):** `force_eval_extinction_gray_zone`; scheduler YAML; task list `Q1_H3_GRAY_ZONE_CONFIRMATORY.md`. Historical D1@0.45 **retired as gate** on production filter. | Locked production H3 blocked; confirmatory duplicate tier optional |
| 2026-07-29 | **Gray-zone exploratory pilot DONE** ($n{=}10$): Holm/NI in `H3_GRAY_HOLM.md`; skip 11.7%; mean $\Delta$cov $-0.52$ pp | Exploratory pass $\neq$ confirmatory unlock; prod.\ filter not rehabilitated |
| 2026-07-31 | **Gray-zone confirmatory DONE** ($n{=}10$; tier `q1-v3-h3-gray-zone`): Holm/NI in `H3_GRAY_HOLM.md`; skip 11.9%; mean $\Delta$cov $-0.60$ pp (2/10 wins) | F-RQ3-gray pass; prod.\ filter not rehabilitated |
| 2026-07-31 | **Matched H1 multi-provider ($n{=}10$):** gpt-4o-mini paired-$t$ TOST $\pm$2 pp accept; DeepSeek mean-TOST reject (90\% CI $[-1.01,3.06]$; bootstrap-median accepts as different estimand) | Does not amend confirmatory Holm; exploratory replication |
| 2026-08-01 | **Cross-stack confound disclosure:** H1/H2 reframed as two case studies (not attach-point $2\times2$ under LLM). Protocol [`Q1_H1H2_MIXED_STACK_2X2.md`](Q1_H1H2_MIXED_STACK_2X2.md); `llm.stub_hints_only` + `map_elites_scheduler_nightly_llm_filter_stub.yaml` for missing cell C | Isolation grid not yet run; manuscript claims narrowed |
| 2026-07-17 | **B3 standard benchmarks complete (§3.12):** deterministic D=20 Sphere/Rastrigin runner, CLI, batch tiers, traces, and benchmark-safe aggregation wired; five smoke combinations passed; **50/50** full runs complete. Sphere CMA-ME beats ME random by **+50.03 pp coverage** and **+290,872 QD-score** (10/10; descriptive p=0.00098, A₁₂=1.00). Rastrigin CMA-ME reaches **90.35%** coverage vs CMA-MAE **32.17%**. Supplementary implementation sanity only; confirmatory families unchanged. |

| 2026-07-30 | **Supplementary Sphere H2:** `me_uniform` vs `me_filter` on Fontaine Sphere (D=20); offline sklearn MLP + threshold gate + empty-bin force-eval; n=10; 32500 proposals; descriptive matched-eval +7.88 pp (10/10). Artifacts: `artifacts/experiments/q1-v3-sphere-h2/`. Does not amend confirmatory Holm families; no LLM/H1. |
