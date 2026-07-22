# Q1 experiment protocol v2 (grid primary + CVT sensitivity)

Operational runbook for a publication-grade comparison: **stub hints vs trained surrogate vs acquisition filter**, with a **supplementary CVT archive geometry block**.

Supersedes: [`EXPERIMENT_PROTOCOL_Q1.md`](EXPERIMENT_PROTOCOL_Q1.md) for planning; v1 remains valid for the **grid primary** experiment.

Related: [`docs/SURROGATE_MODEL.md`](../docs/SURROGATE_MODEL.md), [`docs/MAPELITES.md`](../docs/MAPELITES.md) §4.4, [`artifacts/filter_policy_recovery_plan.md`](filter_policy_recovery_plan.md).

---

## 0. Design overview

```text
┌─────────────────────────────────────────────────────────────────┐
│  PRIMARY (grid 50×50, schema 1.2) — main paper claims           │
│  q1-full: 10 seeds × stub + hints + filter = 30 runs            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SUPPLEMENTARY (CVT 2500 niches, schema 1.3) — sensitivity      │
│  q1-cvt:  10 seeds × stub + hints + filter = 30 runs            │
│  (same seeds, same budget, different archive geometry)          │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Archive | Role in paper | Runs |
|-------|---------|---------------|------|
| **Primary** | Grid (`archive_type: grid`, 50×50) | Main results for RQ1–RQ3 | 30 |
| **Sensitivity** | CVT (`archive_type: cvt`, `n_centroids: 2500`) | Confirms conclusions are not grid-artefacts | 30 |

**Total LLM experiment (primary + CVT):** 60 runs × 6 500 calls = **390 000** API calls at q1-full scale.

---

## 1. Research questions

### 1.1 Primary (grid)

| ID | Question | Conditions |
|----|----------|------------|
| **RQ1** | Do real surrogate hints improve QD archive quality vs fixed stubs? | `stub` vs `hints` |
| **RQ2** | Is the surrogate accurate enough to trust? | holdout on buffer (offline); runtime gate before hints/filter |
| **RQ3** | Can acquisition filter cut sim cost without hurting coverage? | `hints` vs `filter` (operational; see §3.5) |

### 1.2 Supplementary (CVT sensitivity)

| ID | Question | Comparison |
|----|----------|------------|
| **RQ1-s** | Does the stub→hints effect **direction and magnitude** hold under Voronoi niches? | Δ(grid) vs Δ(CVT), same seeds |
| **RQ3-s** | Does filter preserve QD under CVT geometry? | hints vs filter on CVT (exploratory) |

RQ2 is **not re-run** on CVT: the surrogate sees genome features only; archive geometry does not change `predict()`. CVT sensitivity tests **search dynamics and acquisition**, not model accuracy.

**Primary metrics (per seed, both archive types):**

- `coverage` — `filled_cells / n_cells` (2 500 for grid and CVT)
- `mean_best_fitness` — mean fitness over occupied niches
- `evaluations` — real `run_world` count (filter arm lower)
- `skip_rate_pct` — from `surrogate_archive.jsonl` (filter / shadow only)

**Secondary:** elapsed wall time, LLM fallback rate (`emitter_type == llm_fallback`).

---

## 2. Evaluation of the proposed CVT design

### 2.1 Original proposal

> 1. Full set of **30 runs** for RQ1 (Regular vs CVT baseline comparison).
> 2. Reduced set of **30 runs** for one method (Hints or Filter) on CVT.
> 3. Present as supplementary sensitivity confirming main conclusions.

### 2.2 Assessment and correction

| Proposal item | Verdict | Correction |
|---------------|---------|------------|
| 30 runs for RQ1 grid vs CVT | ⚠️ Partially redundant | Grid RQ1 is already covered by primary **q1-full** (stub + hints among 30 runs). **Do not re-run 30 grid arms** for geometry comparison. |
| 30 runs for one CVT method | ⚠️ Under-powered for RQ3-s | Filter-only on CVT (10 runs) cannot validate eval reduction; need hints arm for paired comparison. |
| 30 + 30 = 60 CVT runs | ❌ Double-counts hints | Hints appear in both blocks. |

**Recommended balanced design (adopted in this protocol):**

1. **Primary unchanged:** grid `q1-full`, 10 seeds × (stub, hints, filter) = **30 runs** — all main claims.
2. **CVT sensitivity:** mirror the same matrix on CVT, **10 seeds × (stub, hints, filter) = 30 runs**.
3. **Analysis split (not extra runs):**
   - **RQ1-s:** per-seed Δcoverage(hints − stub) on grid vs CVT (paired by seed).
   - **RQ3-s:** per-seed Δeval, Δcoverage(filter − hints) on grid vs CVT.

This delivers the scientific intent (geometry sensitivity + confirmation of main findings) at **30 CVT runs**, not 60.

**If budget is tight** (minimal CVT supplement):

| Tier | CVT runs | Seeds × conditions | When |
|------|----------|-------------------|------|
| `q1-cvt-min` | 20 | 10 × (stub, hints) | RQ1-s only |
| `q1-cvt` | 30 | 10 × (stub, hints, filter) | RQ1-s + RQ3-s (recommended) |

**Paper framing:**

> *"The primary experiment uses a uniform 50×50 behaviour-characteristic grid. We replicate the full condition matrix on a CVT archive with 2 500 Voronoi niches (same BC axes, same simulation budget) as a **sensitivity analysis**. We report whether the direction and approximate magnitude of surrogate-hint and filter effects are consistent across archive geometries; CVT results are not used for primary hypothesis tests."*

---

## 3. Prerequisites

### 3.0 Production checkpoint (grid + CVT)

**Canonical artifacts** (Jun 2026):

| Artifact | Path |
|----------|------|
| Checkpoint | `artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl` |
| Training summary | `artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json` |
| Uncertainty calibration | `artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl` |

Trained with **MC-dropout** (`dropout_p=0.05`, `uncertainty_method=ensemble_mc`, `mc_samples=16`) so acquisition policy sees **non-constant** calibrated uncertainty (v2 ensemble-only spread was near-constant and collapsed the uncertainty branch of `threshold_gate`).

Symlinks `nightly_v2.pkl`, `latest.pkl`, `calibration.pkl` → above files for backward compatibility.

### 3.0.1 LLM stack v2 (Jun 2026)

Primary Q1 LLM runs use **stack v2** only:

| Component | v1 (deprecated) | v2 |
|-----------|-----------------|-----|
| Model | `qwen-plus` | `qwen-turbo` |
| User prompt | `reasoning` + `world_spec` | `world_spec` only |
| `max_tokens` | 500 | 350 (`llm_world_generator_qwen.yaml`; logged) |
| Parallel LLM emit | sequential | `llm_parallel_emit: true` (workers = LLM slots/batch) |

- Do **not** mix v1 and v2 runs in `q1-full` / `q1-cvt` matrices.
- v1 shadow hints seed 0 archived under `artifacts/experiments/shadow/v1_backup_2026-06-28/`.
- Archive `metadata.prompt_version` = `{system_hash}:{user_hash}` (composite); system-only hashes are v1-era.
- Operational plan: [`Q1_LLM_STACK_V2_T0_T2_PLAN.md`](Q1_LLM_STACK_V2_T0_T2_PLAN.md), [`Q1_LLM_SPEEDUP_PLAN.md`](Q1_LLM_SPEEDUP_PLAN.md).

#### API stack pin (`qwen-turbo` is not versioned)

DashScope serves `qwen-turbo` as a **floating alias** (backend weights can change without a new model string). Q1 did **not** call a dated snapshot ID (e.g. `qwen-turbo-2025-04-28`). Reproducibility of the remote stack is therefore **window-pinned**, not weight-pinned:

| Pin | Value |
|-----|-------|
| Provider / region | Alibaba Cloud DashScope **intl** (Singapore-compatible endpoint) |
| `api_base` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions` |
| Request `model` | **`qwen-turbo`** (floating alias — as logged in all `nightly_run_summary.json`) |
| Spec file | `worldspace/specs/llm_world_generator_qwen.yaml` (`llm_spec_hash=5f87dea8bbef458a` on post-§3.0.2 runs) |
| Auth | `QWEN_API_KEY` env |
| Observation window (artifact mtimes) | **2026-07-02 → 2026-07-11** |
| Matrix coverage in window | `q1-full` 2026-07-02 · `q1-cvt` 2026-07-09 · `q1-repeat` + `q1-prompt-ablation` 2026-07-11 |
| Decoding | `temperature=0.2`; `top_p` omitted (provider default); no client API `seed` |

**Disclosure for paper / reviewers:** results are conditional on whatever backend DashScope bound to `qwen-turbo` during that window. Exact weight snapshot IDs were **not** returned in our logs and **cannot be recovered retroactively**. Provider docs list dated IDs such as `qwen-turbo-2025-04-28` / `qwen-turbo-2024-11-01` for future pins; migrating the YAML to a dated ID is **out of Q1 scope** (would invalidate the matrix).

**Future hardening:** set `model: qwen-turbo-YYYY-MM-DD` (or successor `qwen-flash-…`) in the LLM spec; log request `model` + response `model` fields in `nightly_run_summary.json`.

### 3.0.2 LLM stochasticity and variance floor

Remote `qwen-turbo` calls are **not fully reproducible** even with fixed MAP-Elites `seed`:

| Control | Value | Logged in `nightly_run_summary.json` |
|---------|-------|--------------------------------------|
| `temperature` | **0.2** (`llm_world_generator_qwen.yaml`) | `llm_temperature` |
| `top_p` | provider default (not set in YAML) | `llm_top_p` (null) |
| API `seed` | not supported in client | — |
| Spec pin | `llm_world_generator_qwen.yaml` | `llm_spec_path`, `llm_spec_hash` |
| API endpoint | DashScope intl compatible-mode (see §3.0.1) | fixed in YAML; not duplicated in summary |
| Model snapshot | **floating** `qwen-turbo` (no dated ID) | window **2026-07-02…2026-07-11** only |

**Paired comparisons** (stub vs hints, hints vs filter) use the **same MAP-Elites seed** but **independent LLM rollouts** (6500 calls each). Stochasticity is therefore **between-run**, not paired at the token level.

**Variance floor (measurement noise):** before interpreting borderline Δ (especially RQ3 coverage near −3 pp), run **`q1-repeat`**:

```bash
./scripts/run_experiment_batch.sh q1-repeat 0 2   # recommended: ≥3 seeds
# minimal: ./scripts/run_experiment_batch.sh q1-repeat 0 1
```

Matrix (recommended): **≥3 seeds × (stub, hints) × 3 replicates**. Analysis (`scripts/analyze_q1_statistics.py`) computes:

```text
floor[metric] = 2 × pooled_within_sd(replicates per condition×seed)
```

**Caveat:** with only a few condition×seed groups the floor point estimate is **noisy**. Use it as a coarse scale check for large effects (e.g. RQ1 ~15 pp ≫ floor), **not** as a sharp threshold for differences of order ~1 pp (seed-level diagnostics, 1-seed ablation deltas). Do not treat `diff < floor` as a formal test.

Any confirmatory |median Δ| clearly below `floor` → `noise_indistinguishable` (interpret with caution).

**Fresh pair (shadow only):** hints and filter/shadow for archive parity must still be run in **one batch session** (§8.6); this is separate from the repeat floor.

### 3.1 RQ2: offline validation + runtime quality gate

**Two tiers** (see `worldspace/surrogate/evaluation.py`, `checkpoint_quality.py`):

| Tier | Summary field | Hold-out thresholds | Used for |
|------|---------------|---------------------|----------|
| **Pilot / runtime gate** | `hints_ok` | R²(fitness) ≥ 0.30, MAE(fitness) < 0.085 | `--require-surrogate-quality-gate` on hints/filter arms |
| **Production / RQ2 report** | `quality_passed` | R²(fitness) > 0.72, MAE(fitness) < 0.085, MAE(stability) < 0.06 | Paper Table 3, checkpoint write policy |

**Runtime behaviour:** when `require_quality_gate` is set (default in `run_experiment_batch.sh` for hints and filter), `get_surrogate()` calls `checkpoint_quality_allows_hints()` — **`hints_ok` is sufficient**; `quality_passed` is only a legacy fallback when `hints_ok` is absent. If the gate fails, the facade becomes `StubSurrogate` → **hints arm ≡ stub arm** in the LLM prompt (fixed `stub_mean` / `stub_uncertainty`).

**What R²(fitness) measures (D2 — single-seed labels):** each buffer row is **one** `evaluate_candidate` at `canonical_seed(WorldSpec)` (stochastic CA via genome `noise`; not MC-averaged). The nightly buffer (`buffer_nightly.jsonl`) stores **component targets only** (no `fitness` column) → checkpoint has **no direct fitness head** (`_has_fitness_head: false`). Hold-out R²/MAE in `evaluate_holdout` compare **composed** fitness from predicted vs held-out components (`compute_fitness_from_prediction`, gate **0.5** default). This is **not** \(E[\text{fitness} \mid \text{WorldSpec}]\) and **not** identical to archive fitness from simulation `early_extinct` (see §3.6 D1). See §3.6 for paper captions.

**Exit criteria before q1-full:**

1. `hints_ok: true` in summary (required for real hints/filter at runtime).
2. `quality_passed: true` in summary (required for RQ2 production claim in the paper).

**Verification (do not use `compare_acquisition_runs.py` for this):**

```bash
jq '{hints_ok, quality_passed, holdout_metrics}' \
  artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json
```

Current checkpoint (Jun 2026): both flags `true` (R² ≈ 0.76, MAE(fitness) ≈ 0.007, MAE(stability) ≈ 0.04).

If `hints_ok` is false → hints/filter runs are stub-equivalent; fix checkpoint or train with `--no-quality-gate` only for debugging, not for publication arms.

### 3.2 Shared setup (grid + CVT)

```bash
# Grid baseline + surrogate buffer (~32.5k sims, 0 LLM) — already done for primary
uv run python -m worldspace.scripts.run_map_elites_nightly --archive-type grid

# Verify surrogate checkpoint (RQ2 offline — optional cross-check)
uv run python scripts/analyze_surrogate_buffer.py \
  --buffer artifacts/surrogate/buffer_nightly.jsonl \
  --fit-model --output-json artifacts/surrogate/buffer_analysis.json

# (Re)train + calibrate if checkpoint missing — MC-dropout recipe
uv run python scripts/train_surrogate.py \
  --buffer-path artifacts/surrogate/buffer_nightly.jsonl \
  --checkpoint-path artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl \
  --summary-path artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json \
  --mlp-dropout-p 0.05 \
  --mlp-uncertainty-method ensemble_mc \
  --mlp-mc-samples 16 \
  --calibrate \
  --calibration-path artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl \
  --acquisition-report \
  --no-quality-gate
```

`run_experiment_batch.sh` trains the same recipe automatically when the checkpoint file is absent.

### 3.3 CVT-specific (required before `q1-cvt`)

```bash
# CVT baseline archive (0 LLM, same iter/batch budget as grid)
uv run python -m worldspace.scripts.run_map_elites_nightly --archive-type cvt
```

Expected artifacts:

| Archive type | Baseline JSONL | Centroids |
|--------------|----------------|-----------|
| Grid | `artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl` | — |
| CVT | `artifacts/map_elites_nightly/cvt/baseline/map_elites_archive.jsonl` | `.../cvt_centroids.json` |

**Resume rule:** each LLM run loads the baseline matching its scheduler `archive.type`. Grid schedulers → grid baseline; CVT schedulers → CVT baseline. **Never cross-load** (different niche geometry).

### 3.4 Scheduler YAMLs to add (not yet in repo)

Fork nightly grid specs into CVT (`schema_version: "1.3"`, `archive.type: cvt`, `n_centroids: 2500`, `cvt_seed: 0`):

| Condition | Planned YAML |
|-----------|--------------|
| stub | `worldspace/specs/map_elites_scheduler_nightly_llm_stub_cvt.yaml` |
| hints | `worldspace/specs/map_elites_scheduler_nightly_llm_cvt.yaml` |
| filter | `worldspace/specs/map_elites_scheduler_nightly_llm_filter_cvt.yaml` |
| shadow | `worldspace/specs/map_elites_scheduler_nightly_llm_shadow_cvt.yaml` |

Template: `map_elites_scheduler_github_llm_cvt.yaml` (pilot shape) + nightly filter/shadow policy blocks from grid specs.

**CVT LLM prompt:** system prompt `prompts/map_elites_llm_emitter_system_cvt.txt` (Voronoi / `cell_id` wording). Surrogate checkpoint and filter thresholds: **start with grid-tuned values**; run **one CVT shadow seed** before `q1-cvt` filter arm if live skip rate falls outside 25–45%.

### 3.5 Grid filter calibration (before primary `q1-full`)

Policy: `threshold_gate` — skip only when **both** `fitness < min_predicted_fitness` **and** `uncertainty ≤ max_uncertainty_to_skip`, plus `never_skip_empty_bin: true` in live runs.

Grid filter YAML (`map_elites_scheduler_nightly_llm_filter.yaml`):

- `extinction_gate_threshold: 0.95` — **active**: runtime composed fitness zeros when predicted `early_extinction_prob ≥ 0.95` (no fitness head in `nightly_v3_mc_d005`). See §3.6 (D1).
- `min_predicted_fitness: 0.45` — **Q1 combat / `q1-full` value** (raised from **0.10** after shadow seed_0: 0.10 → ~12% skip; 0.45 → ~34% in band). Logged filter skips match `fitness < 0.45` (agree=1.0).
- `max_uncertainty_to_skip: 1.0`
- `target_selection: uniform_frontier`

**Shadow calibration** (mandatory before q1-full):

```bash
export QWEN_API_KEY=...
# Remove stale seed dirs if re-calibrating after checkpoint change:
# rm -rf artifacts/experiments/shadow/hints/seed_0 artifacts/experiments/shadow/filter/seed_0
./scripts/run_shadow_calibration.sh 0
```

Script steps: (1) offline buffer replay, (2) live hints + shadow pair seed 0, (3) `compare_acquisition_runs.py`.

| Metric | Source | Target / interpretation |
|--------|--------|-------------------------|
| `hints_ok` / `quality_passed` | `nightly_v3_mc_d005.summary.json` | §3.1 exit criteria |
| `false_skip_rate_estimate` | offline replay (step 1) | < 5% |
| `recommended_skip_rate` | offline replay | **upper bound only** (~90%+); replay uses empty 10×10 grid and `never_skip_empty_bin=false` |
| **Live `shadow_would_skip`** | `surrogate_archive.jsonl` in shadow run (step 3) | **25–45%** (primary tuning target) |
| Archive parity | `compare_acquisition_runs` hints vs shadow | **coverage and mean fitness must match** (shadow always evaluates; archives identical if same seed/checkpoint/prompt path) |

**Do not** treat offline ~97% skip as the filter-arm expectation. **Do not** use `compare_acquisition_runs` to read `quality_passed`.

If live shadow skip falls outside 25–45%, retune `min_predicted_fitness` / `max_uncertainty_to_skip` and re-run shadow (one seed) before `q1-full`. See [`filter_policy_recovery_plan.md`](filter_policy_recovery_plan.md).

**Decision card (B1 + uncertainty vs fitness-gate tracks):** [`DECISION_CARD_B1_UNCERTAINTY_ACQUISITION.md`](DECISION_CARD_B1_UNCERTAINTY_ACQUISITION.md).

**RQ3 framing:** RQ3 tests whether the **filter arm** reduces evaluations without unacceptable QD loss vs hints. It does **not** require claiming uncertainty-driven acquisition unless live shadow shows a non-trivial `HIGH_UNCERTAINTY_FORCE_EVAL` rate in `surrogate_archive.jsonl`.

### 3.6 Methodological risks: extinction compose (D1) and single-seed labels (D2)

Tracked items with explicit exit criteria.

#### D2 — single-seed training targets (closed by documentation)

| Item | Status |
|------|--------|
| Risk | One stochastic CA rollout per `WorldSpec` at canonical seed; component targets are not MC-averaged. |
| RQ2 metric | Hold-out R²/MAE score **composed** fitness from single-rollout component labels (buffer has no `fitness` column). |
| Exit | Documented in §3.1 and §10 (Table 3 caption). **No code change.** |

**Paper wording (Table 3 caption / RQ2 text):**

> *Hold-out R²(fitness) and MAE(fitness): MLP ensemble predicting illuminator fitness **recomposed from component targets** on a single canonical stochastic CA realization per genome (seed = hash(WorldSpec)); not an expectation over multiple noise trajectories. Offline hold-out uses compose gate 0.5; live acquisition uses gate 0.95 (§3.6 D1).*

#### D1 — extinction proxy vs archive vs compose gate

| Layer | Extinction rule |
|-------|-----------------|
| **Archive (simulation)** | Binary `early_extinct` (density == 0 before step 200) → fitness = 0; else continuous penalty via `(1 − ext_p)`, `ext_p = 1 − final_density`. |
| **Buffer target `early_extinction_prob`** | Always `1 − final_density` — smooth proxy, **not** the simulation `early_extinct` flag. |
| **RQ2 hold-out (`evaluate_holdout`)** | Composed fitness on component targets; hard gate when `early_extinction_prob ≥ 0.5` (code default). |
| **Live hints/filter (`nightly_v3_mc_d005`)** | Composed fitness on **predicted** components; hard gate when `early_extinction_prob ≥ 0.95` (YAML). `use_soft_extinction: false`. No fitness head. |

**Why this matters:** near edge-of-chaos (low `final_density`, not simulation–early-extinct), archive can assign small nonzero fitness while compose-hard may zero predicted fitness — affecting hints/filter skip decisions. Hold-out R² is computed with a **stricter compose gate (0.5)** than runtime **(0.95)**, so Table 3 is not a direct calibration of live gate behaviour.

**D1 exit (documentation path — adopted for Q1):**

1. Disclosed in this section and §8 (limitations).
2. `use_soft_extinction` remains `false` (hard compose, not soft blend).
3. Optional sensitivity checks below; **no checkpoint retrain required** unless compose A/B shows unacceptable divergence on high-occupancy bins.

**Optional sensitivity checks (executed for Q1 report):**

```bash
uv run python scripts/run_d1_compose_checks.py
# → artifacts/surrogate/compose_ab_check.json
# → artifacts/surrogate/compose_gate_0p5_vs_0p95.json   # buffer-holdout proxy (superseded for B.2)

uv run python scripts/replay_compose_gate_live.py
# → artifacts/surrogate/compose_gate_live_0p5_vs_0p95.json  # primary B.2 gate
```

**Measured results — primary B.2 (live proposal replay, `q1-full/filter` seeds 0–9):**

| Check | Result |
|-------|--------|
| Combat threshold | `min_predicted_fitness=0.45` (YAML + logged skips) |
| Gate 0.5 vs 0.95 divergent skip @ combat `min_fit=0.45` | **0.528** ≫ 0.05 (n=325 000 proposals) |
| Gate 0.5 vs 0.95 divergent skip @ `min_fit=0.10` (sensitivity) | **0.746** ≫ 0.05 — **higher**, not lower |
| Pred \(p_{ext} \in [0.5, 0.95)\) | ~0.746 of live proposals |
| Agree logged skip vs recompose @ gate 0.95 | **1.000** @0.45; ~0.78 @0.10 (logs were produced at 0.45) |
| B.2 confirmatory rule (`div ≤ 0.05`) | **FAIL → RQ3 exploratory** at both thresholds |

**Why 0.10 does not rescue confirmatory:** under gate 0.5, band proposals compose to fitness 0 → skip at either threshold; under gate 0.95 many keep fitness ∈ (0.10, 0.45). Lowering `min_fit` therefore **increases** skip divergence (more eval@0.95 / skip@0.5 pairs). Exploratory status is not an artifact of using 0.45 instead of 0.10.

**Superseded proxy (buffer hold-out, n=1500, `nightly_v3_mc_d005`) — not used for B.2:**

| Check | Result |
|-------|--------|
| Hard vs soft compose | hard R²≈0.64, soft R²≈−0.32 → keep hard (`use_soft_extinction: false`) |
| Gate 0.5 vs 0.95 divergent skip @ `min_fit=0.45` | **0.432** ≫ 0.05 (proxy only) |
| Same @ `min_fit=0.10` | **0.700** ≫ 0.05 (proxy only) |
| Pred \(p_{ext} \in [0.5, 0.95)\) | ~70% of hold-out rows |

**Caveat:** primary D1 = live filter `surrogate_archive.jsonl` replay at **combat 0.45**. Buffer proxy and the historical §3.5 listing of 0.10 are **superseded** for the confirmatory gate. Operational filter arms still show eval↓ ~33% with QD non-inferiority — report those as **exploratory / operational**, not confirmatory under the D1 quantitative gate.

**Future hardening (out of Q1 scope):** backfill `fitness` → fitness head; align `evaluate_holdout` gate with YAML 0.95.

---

## 4. Experiment tiers

### 4.1 Primary (grid) — same as v1

| Tier | Script arg | Seeds | Conditions | Iterations | LLM / run | Total LLM |
|------|------------|-------|------------|------------|-----------|-----------|
| Pilot | `pilot` | 3 (0–2) | stub, hints | 120 | 2 400 | 14 400 |
| Q1 minimum | `q1-min` | 10 (0–9) | stub, hints | 650 | 6 500 | 130 000 |
| **Q1 full** | `q1-full` | 10 (0–9) | stub, hints, filter | 650 | 6 500 | **195 000** |
| Q1 robust | manual | 20 (0–19) | stub, hints, filter | 650 | 6 500 | 390 000 |

### 4.2 Supplementary (CVT)

| Tier | Script arg (planned) | Seeds | Conditions | Iterations | LLM / run | Total LLM |
|------|----------------------|-------|------------|------------|-----------|-----------|
| CVT pilot | `cvt-pilot` | 3 (0–2) | stub, hints | 120 | 2 400 | 14 400 |
| CVT minimum | `q1-cvt-min` | 10 (0–9) | stub, hints | 650 | 6 500 | 130 000 |
| **CVT full** | `q1-cvt` | 10 (0–9) | stub, hints, filter | 650 | 6 500 | **195 000** |

**Execution order:**

```text
1. Grid shadow (seed 0)           → validate filter thresholds + archive parity (run_shadow_calibration.sh)
2. Grid q1-full (seeds 0–9)       → primary results
3. CVT baseline nightly           → prerequisite
4. CVT shadow (seed 0, optional)  → if filter skip rate diverges from grid
5. CVT q1-cvt (seeds 0–9)         → sensitivity (same seeds as step 2)
```

---

## 5. Scheduler specs

### 5.1 Grid (primary)

| Condition | YAML | `surrogate.enabled` | Notes |
|-----------|------|---------------------|-------|
| stub | `map_elites_scheduler_nightly_llm_stub.yaml` | `false` | fixed `{0.5, 1.0}` |
| hints | `map_elites_scheduler_nightly_llm.yaml` | `true` | `nightly_v3_mc_d005.pkl` + `--require-surrogate-quality-gate` |
| filter | `map_elites_scheduler_nightly_llm_filter.yaml` | `true` | `acquisition.mode: filter` |
| shadow hints | `map_elites_scheduler_nightly_llm_shadow_hints.yaml` | `true` | acquisition off; parity control for shadow |
| shadow | `map_elites_scheduler_nightly_llm_shadow.yaml` | `true` | `acquisition.mode: shadow`; logs `shadow_would_skip` |

### 5.2 CVT (sensitivity)

| Condition | YAML (to create) | Notes |
|-----------|------------------|-------|
| stub | `..._nightly_llm_stub_cvt.yaml` | same surrogate block as grid stub |
| hints | `..._nightly_llm_cvt.yaml` | CVT system prompt |
| filter | `..._nightly_llm_filter_cvt.yaml` | transfer grid thresholds initially |
| shadow | `..._nightly_llm_shadow_cvt.yaml` | 1-seed calibration |

Shared CVT archive block:

```yaml
schema_version: "1.3"
archive:
  type: cvt
  n_centroids: 2500
  cvt_seed: 0
  lloyd_iterations: 50
```

Filter/shadow arms should also set `target_selection: uniform_frontier` (same rationale as grid).

---

## 6. Run matrix

### 6.1 Primary batch (grid)

```bash
export QWEN_API_KEY=...

# Shadow calibration (once per checkpoint / threshold change)
# Ensures fresh hints+shadow pair; delete existing seed_0 dirs if resuming stale runs.
./scripts/run_shadow_calibration.sh 0

# Primary experiment
./scripts/run_experiment_batch.sh q1-full 0 9
```

Output: `artifacts/experiments/q1-full/{stub,hints,filter}/seed_*/`

### 6.1.1 LLM variance floor (`q1-repeat`)

Run **after** checkpoint/prompt stack is pinned, **before** or **in parallel with** writing primary claims if reviewers ask about API noise:

```bash
./scripts/run_experiment_batch.sh q1-repeat 0 1
uv run python scripts/analyze_q1_statistics.py   # picks up floor from q1-repeat/summary.csv
```

Output: `artifacts/experiments/q1-repeat/{stub,hints}/seed_*/rep_{0,1,2}/`

Cost: 12 × 6500 ≈ **78k** LLM calls (~2 grid seeds worth of wall time).

### 6.2 CVT batch (planned — extend `run_experiment_batch.sh`)

```bash
# After CVT baseline exists:
./scripts/run_experiment_batch.sh cvt-shadow 0 0    # optional
./scripts/run_experiment_batch.sh q1-cvt 0 9
```

Output: `artifacts/experiments/q1-cvt/{stub,hints,filter}/seed_*/`

Each CVT run directory must contain `cvt_centroids.json` (copied or produced at run start).

### 6.2.1 Prompt ablation (optional; recommend ≥3 seeds)

CVT archive geometry with **grid** system prompt (disentangle wording from Voronoi niches).

**Minimum (done):** seed 0 — descriptive sanity check.  
**Recommended before strong claims:** seeds **0–2** (3× stub+hints ≈ **39k** LLM calls) so the ablation Δ can be compared across seeds without leaning on the noisy q1-repeat floor point estimate.

```bash
./scripts/run_experiment_batch.sh q1-prompt-ablation 0 2   # recommended
# already have seed 0; resume skips completed dirs
```

Output: `artifacts/experiments/q1-prompt-ablation/{stub,hints}/seed_*/`

Compare per-seed Δcoverage(hints−stub) to matching `q1-cvt` seeds. Report sign concordance and mean |Δ_ablation − Δ_cvt|; treat single-seed |diff| vs floor as **illustrative only**.

### 6.3 Single run (CVT hints example)

```bash
uv run python scripts/run_github_llm_map_elites.py \
  --scheduler worldspace/specs/map_elites_scheduler_nightly_llm_cvt.yaml \
  --output-dir artifacts/experiments/q1-cvt/hints/seed_3 \
  --seed 3 \
  --iterations 650 \
  --load-archive artifacts/map_elites_nightly/cvt/baseline/map_elites_archive.jsonl \
  --llm-provider qwen \
  --require-surrogate-quality-gate
```

`run_github_llm_map_elites.py` resolves baseline from scheduler `archive_type` when `--load-archive` is omitted.

### 6.4 Aggregation

```bash
# Primary
uv run python scripts/aggregate_experiment_runs.py \
  --root artifacts/experiments/q1-full \
  --output artifacts/experiments/q1-full/summary.csv

# CVT sensitivity
uv run python scripts/aggregate_experiment_runs.py \
  --root artifacts/experiments/q1-cvt \
  --output artifacts/experiments/q1-cvt/summary.csv
```

### 6.5 Cross-geometry comparison (RQ1-s / RQ3-s)

Per seed, join grid and CVT summaries:

```text
seed_0:  Δcov_grid  = cov_hints - cov_stub   (grid)
         Δcov_cvt   = cov_hints - cov_stub   (cvt)
         consistency: sign(Δcov_grid) == sign(Δcov_cvt)
```

Acquisition A/B within each geometry (`compare_acquisition_runs.py` — **RQ3 metrics only**, not RQ2):

```bash
# Grid RQ3 (primary): eval reduction, coverage delta, live skip rate from shadow/filter archive
uv run python scripts/compare_acquisition_runs.py \
  --baseline-dir artifacts/experiments/q1-full/hints/seed_0 \
  --candidate-dir artifacts/experiments/q1-full/filter/seed_0 \
  --grid-resolution 50

# Shadow parity check (after run_shadow_calibration.sh): hints vs shadow — archives must match
uv run python scripts/compare_acquisition_runs.py \
  --baseline-dir artifacts/experiments/shadow/hints/seed_0 \
  --candidate-dir artifacts/experiments/shadow/filter/seed_0 \
  --grid-resolution 50

# CVT RQ3-s
uv run python scripts/compare_acquisition_runs.py \
  --baseline-dir artifacts/experiments/q1-cvt/hints/seed_0 \
  --candidate-dir artifacts/experiments/q1-cvt/filter/seed_0 \
  --grid-resolution 50
```

---

## 7. Statistical analysis plan

### 7.1 Primary (grid)

Analyze **per-seed aggregates**. Wilcoxon signed-rank, bootstrap 95% CI, Cohen's `d`.

**RQ1 pass (hints vs stub):** mean Δcoverage > 0 with CI excluding 0, or mean Δfitness > 0 with p < 0.05 and d ≥ 0.5.

**RQ3 pass (filter vs hints) — amended 2026-07-11:**

| Gate | Confirmatory criterion | Unit of analysis |
|------|------------------------|------------------|
| Eval reduction | median relative Δevals(filter−hints) ≤ −20%; one-sided Wilcoxon | paired seed deltas (n=10) |
| Coverage QD | **non-inferiority:** one-sided test that mean Δcoverage(filter−hints) **> −3 pp** | paired seed deltas (n=10) |
| Fitness QD | **non-inferiority:** one-sided test that mean relative Δfitness(filter−hints) **> −5%** | paired seed deltas (n=10) |

**Confirmatory vs descriptive (do not mix):**

- **Confirmatory** = the §7 tests on the **vector of 10 paired seed-level Δ** (Wilcoxon / TOST / non-inferiority + Holm where applicable). Pass/fail for RQ3 coverage and fitness is decided **only** here.
- **Not confirmatory:** per-seed maximum `|Δcoverage|`, worst-seed Δ, or “every seed within ±3 pp”. Those are **descriptive diagnostics** (e.g. seed 6 Δcov = −3.32 pp). A single seed outside the operational band does **not** overturn a passing mean-level non-inferiority / TOST result.
- Legacy phrases like `|Δcoverage| ≤ 3 pp` meant an operational band around typical loss, **not** a max-over-seeds hard gate. Prefer the explicit NI wording above.

RQ3 asks whether the filter cuts evals **without hurting QD**. Symmetric TOST (`|Δ| ≤ margin`) is the wrong estimand: it rejects when the filter **improves** fitness. The amendment replaces symmetric equivalence with one-sided non-inferiority aligned with that scientific claim.

**Transparency (dual reporting):** analyses still compute the pre-amendment **symmetric TOST** family (Holm m=4 including `RQ3_cov_TOST` / `RQ3_fit_TOST`) and report it as `RQ3_formal_TOST`. The **paper / primary claim** uses `RQ3_amended_noninferiority`. Mark this change as **post-hoc** relative to the locked confirmatory family. Formal TOST for coverage also operates on the paired-Δ vector (mean), not on max-over-seeds.

**Legacy (pre-amendment) wording:** eval reduction ≥ 20%; `|Δcoverage| ≤ 3 pp`; `|Δfitness| ≤ 5%` relative — interpreted as mean-level operational targets, never as max-over-seeds gates.

**Variance floor guard:** when `q1-repeat` is available, apply §3.0.2 `noise_indistinguishable` before declaring PASS on RQ1/RQ3 superiority arms.

### 7.2 CVT sensitivity (supplementary)

**Do not pool grid and CVT rows for primary hypothesis tests.**

| Analysis | Method | Purpose |
|----------|--------|---------|
| RQ1-s consistency | Per seed: compare sign and rank of Δ(hints−stub) on grid vs CVT | Qualitative confirmation |
| RQ1-s magnitude | Scatter: Δcoverage grid vs CVT (10 points) | Show effect is not grid-specific |
| RQ3-s | Same pass criteria as RQ3 on CVT arm only | Exploratory; report alongside grid |
| Concordance | % seeds where hint benefit (or filter savings) agrees in sign | Simple summary stat |

**Sensitivity pass (for supplementary section):**

- ≥ **8 / 10** seeds show the same sign for Δcoverage(hints − stub) on grid and CVT, **and**
- CVT RQ3-s meets the same eval-reduction / **non-inferiority** QD bounds as grid (§7.1), **or** deviations are documented with CVT-specific skip-rate diagnostics.

---

## 8. Known limitations (disclose in paper)

1. **Surrogate trained on grid-era buffer** — features are genome-only; CVT does not change predictions, only niche assignment and `never_skip_empty_bin` behaviour.
2. **Filter thresholds tuned on grid shadow** — CVT may need separate shadow if skip rate diverges (> 10 pp from grid).
3. **Different LLM system prompts** — CVT uses `map_elites_llm_emitter_system_cvt.txt` (Voronoi wording); grid uses the rectangular-bin prompt. This confounds geometry with prompt text. Acceptable for sensitivity; **not** a pure geometry ablation. Optional check: `q1-prompt-ablation` (CVT archive + **grid** system prompt). Prefer **≥3 seeds** (§6.2.1); seed-0-only is descriptive. Do not adjudicate ablation vs CVT using the q1-repeat floor point estimate alone.
4. **CVT baseline coverage may differ** from grid baseline at the same sim budget — compare arms **within** each archive type, not absolute coverage across types.
5. **Acquisition policy** — `threshold_gate` combines predicted fitness and calibrated uncertainty; with `never_skip_empty_bin` and partial archive coverage, live skip rate is much lower than offline buffer replay. Do not claim uncertainty-driven filtering unless shadow logs show material `HIGH_UNCERTAINTY_FORCE_EVAL` rates.
6. **LLM non-determinism + floating model alias** — see §3.0.1–§3.0.2. Remote API sampling (`temperature=0.2`, no `top_p`/API seed) injects between-run variance. Request model was floating **`qwen-turbo`** (DashScope intl); Q1 is **window-pinned** to **2026-07-02…2026-07-11**, not weight-pinned — dated snapshot IDs were not used and cannot be recovered from logs. **`q1-repeat`** estimates a coarse measurement floor (≥3 seeds recommended); the point estimate has wide uncertainty when based on few groups — use only for large-effect checks. Log `llm_temperature`, `llm_top_p`, `llm_spec_hash` per run. **Fresh pair** (hints+shadow/filter in one batch) is still required for archive parity — do not reuse `Skip existing` dirs across sessions.
7. **Single-seed surrogate labels (D2)** — component targets from one canonical noisy CA realization per genome; R²(RQ2) is on composed fitness, not MC expectation; see §3.6.
8. **Extinction compose vs archive (D1)** — buffer `early_extinction_prob` is a density proxy; live acquisition uses compose-hard at gate **0.95**; hold-out RQ2 uses gate **0.5**; archive uses simulation `early_extinct`; see §3.6.

---

## 9. Cost / time estimates

Assumptions: ~140 ms/sim, ~2 s/LLM call (protocol default); **observed v1** ~23 h/seed (sequential `qwen-plus` + reasoning prompt). Stack v2 (prompt + turbo, still sequential HTTP): ~12–18 h/seed indicative; with parallel emit (T3): ~4–6 h/seed.

| Block | Runs | LLM calls | Wall time (serial, indicative) |
|-------|------|-----------|--------------------------------|
| Grid q1-full | 30 | 195 000 | ~35 h sim + ~108 h LLM |
| Grid q1-repeat | 12 | 78 000 | ~0.4× one q1-full seed block |
| CVT q1-cvt | 30 | 195 000 | ~35 h sim + ~108 h LLM |
| **Combined v2** | 60 | **390 000** | ~2× q1-full |

CVT baseline (no LLM): ~9 h sim once.

Parallelize across seeds (10 workers → ~1–2 days per block).

---

## 10. Paper figure / table mapping

| Output | Figure / table | Section |
|--------|----------------|---------|
| `q1-full/summary.csv` | Table 1: coverage, fitness, evals (grid) | Primary |
| Per-seed Δ stub→hints (grid) | Figure 2: paired dot plot | Primary |
| `compare_acquisition_runs` grid | Table 2: eval reduction vs QD loss | Primary |
| `nightly_v3_mc_d005.summary.json` | Table 3: surrogate validation (RQ2); caption per §3.6 D2 + D1 gate caveat | Primary |
| Δ stub→hints: grid vs CVT scatter | Figure S1 | Supplementary |
| `q1-cvt/summary.csv` | Table S1: CVT arm summaries | Supplementary |
| [`Q1_GRID_CVT_ANALYSIS.md`](Q1_GRID_CVT_ANALYSIS.md) | Grid vs CVT paired analysis (RQ1-s, RQ3-s) | Supplementary |
| CVT archive scatter (best seed) | Figure S2 | Supplementary |

---

## 11. Reproducibility checklist

Status key: **[x]** done with evidence · **[~]** partial / caveat · **[ ]** open.

### LLM stack & logging

- [x] Pin `llm_stack_version: v2`, model `qwen-turbo`, and composite `prompt_version` in run summaries (§3.0.1) — **2026-07-02** grid `q1-full`: `v2` / `qwen-turbo` / `2fc7bdc1:e2afd1e9`; **2026-07-09** CVT `q1-cvt`: same stack, Voronoi prompt `db1dbabf:e2afd1e9`
- [x] Pin `qwen-turbo` API stack by **endpoint + observation window** (§3.0.1) — `api_base=dashscope-intl…/compatible-mode/v1/chat/completions`; floating alias (no dated snapshot); window **2026-07-02→2026-07-11**; disclose: weights not recoverable
- [~] Confirm `llm_temperature`, `llm_top_p`, `llm_spec_hash` logged per LLM run (§3.0.2) — **present** on `q1-repeat` / `q1-prompt-ablation` (**2026-07-11**): `temperature=0.2`, `top_p=null`, `llm_spec_hash=5f87dea8bbef458a`; **absent** on earlier `q1-full` / `q1-cvt` summaries (predates logging)
- [x] Do not mix v1 (`qwen-plus` + reasoning prompt) and v2 runs in one experiment matrix — matrix is v2-only; v1 shadow archived under `artifacts/experiments/shadow/v1_backup_2026-06-28/`

### Surrogate checkpoint & gates

- [x] Pin `nightly_v3_mc_d005.pkl` and `nightly_v3_mc_d005.summary.json` hashes (shared checkpoint) — sha256₁₂ `2efbe19e894a` / `4362b6de43b3`
- [x] Pin `calibration_v3_mc_d005.pkl` hash (filter/shadow arms) — sha256₁₂ `f10ad44614e3`
- [x] Confirm `hints_ok` and `quality_passed` in summary before q1-full (§3.1) — both **True** in `nightly_v3_mc_d005.summary.json`
- [x] D2: Table 3 / RQ2 text uses single-seed + composed-fitness caveat (§3.6) — documented in §3.1 / §3.6 / §10 caption note
- [x] D1: disclose compose gate 0.5 (hold-out) vs 0.95 (runtime) and proxy-vs-simulation extinction (§3.6) — §3.6 + analysis B.2
- [x] D1: run `scripts/run_d1_compose_checks.py`; record compose A/B + buffer gate proxy — **2026-07-12**; hard R²≈0.64 vs soft ≈−0.32; buffer div@0.45=**0.432** (superseded)
- [x] D1: run `scripts/replay_compose_gate_live.py` @ combat **0.45** (div **0.528**) and sensitivity **0.10** (div **0.746**); RQ3 exploratory — **2026-07-12**; n=325 000 live proposals

### Calibration, primary matrix, variance floor

- [x] Complete grid shadow seed 0 with archive parity before filter arm (§3.5) — shadow seed_0: live skip **35.3%** (in 25–45% band); raised `min_predicted_fitness` **0.10→0.45**; shadow dirs under `artifacts/experiments/shadow/` (+ `v2_backup_20260626/`)
- [x] (Recommended) Run `q1-repeat` and record variance floor in [`Q1_GRID_CVT_ANALYSIS.md`](Q1_GRID_CVT_ANALYSIS.md) §7 — **2026-07-11**; seeds 0–2 × stub+hints × 3 reps (18 runs); floor coverage **1.686 pp**, fitness **0.0138**, evals **0.0** (coarse; not for ~1 pp adjudication)
- [x] Use **identical seeds** (0–9) across grid and CVT for paired sensitivity — `q1-full` (**2026-07-02**) and `q1-cvt` (**2026-07-09**) both seeds **0–9**, 30 runs each
- [x] Do not mix pilot (120 iter, 20 LLM slots) with nightly-scale tables — reported tables use 650-iter nightly arms only
- [x] RQ3: dual-report formal TOST + amended non-inferiority; paper uses amended (§7.1, 2026-07-11) **as exploratory if D1 fails** — formal TOST **FAIL** (fit); amended NI **PASS**; both labeled **EXPLORATORY** (D1 live div=0.528>0.05); see analysis §7 B.5b / B.6
- [x] (Optional) Expand `q1-prompt-ablation` to seeds 0–2 before strong prompt-confound claims (§6.2.1) — **2026-07-11**; seeds 0–2; mean Δcov **+21.2** ≈ CVT **+21.6** (3/3 sign; mean |gap| 0.48 pp); analysis §8

### CVT archive provenance

- [x] Record `cvt_seed: 0`, `lloyd_iterations: 50`, `n_centroids: 2500` — pinned in all `*_cvt*.yaml` schedulers used for `q1-cvt` / ablation
- [x] Store `cvt_centroids.json` per CVT run directory — present for all 30 `q1-cvt` runs (`stub`/`hints`/`filter` × seeds 0–9); `n=2500`
- [x] Document whether CVT filter used grid-transferred or CVT-recalibrated thresholds — **grid-transferred** (`min_predicted_fitness: 0.45`, same as grid); YAML header: “grid thresholds transferred initially”; CVT skip ~30.8% still in band (no retune)
- [x] Archive v2 shadow runs under `artifacts/experiments/shadow/v2_backup_*` if comparing across checkpoint generations — `v2_backup_20260626/` present
- [ ] Log git commit SHA for grid and CVT batches separately — **open**: `nightly_run_summary.json` does not currently record git SHA; recover from batch logs / shell history if needed for camera-ready

---

## 12. Quick decision tree

```text
Budget ~200k LLM        → grid q1-full only (v1 sufficient for publication)
Budget ~280k LLM        → q1-full + q1-repeat (variance floor for reviewers)
Budget ~400k LLM        → v2: q1-full + q1-cvt (recommended balanced design)
Budget ~270k LLM        → q1-full + q1-cvt-min (RQ1-s only, skip CVT filter)
Grid effect |Δcov| < 1pp → q1-robust (20 seeds) before any CVT work
CVT infra missing       → run nightly --archive-type cvt; add *_cvt.yaml schedulers
```

---

## 13. Relation to v1

| Topic | v1 | v2 |
|-------|----|----|
| Primary archive | Grid only | Grid (unchanged) |
| CVT | Out of scope | 30-run sensitivity mirror |
| Total runs at full scale | 30 | 60 |
| Main hypotheses | RQ1–RQ3 on grid | Same; CVT is supplementary |
| Proposed "30+30 CVT" | — | Corrected to **30 CVT** (avoid double-count) |

**Successor:** publication baselines / multi-LLM / pyribs — [`EXPERIMENT_PROTOCOL_Q1_v3.md`](EXPERIMENT_PROTOCOL_Q1_v3.md) (draft stub; freeze before new runs).
