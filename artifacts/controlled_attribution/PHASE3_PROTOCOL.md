# Phase 3 protocol draft — controlled attribution confirmatory core

**Date:** 2026-09-02  
**Protocol ID:** `ca-phase3-v0`  
**Programme:** `artifacts/CONTROLLED_ATTRIBUTION_RESEARCH_PROGRAM.md`  
**Draft hash file:** `phase3_protocol_draft.json`  
**How to read:** `PHASE3_FOUNDATION.md`

**Freeze status:** `DRAFT — stamp package ready`. `freeze_in_force` remains false until both are true:

1. an independent registry timestamp (OSF or AsPredicted) names this document's SHA-256;
2. a git tag of the form `phase3-protocol-v0` points at the commit that contains that byte-identical file.

Git history and filesystem mtimes are integrity aids, not registration. Content may still change until that pair exists. After freeze, amendments are dated dual-report only and must not rewrite locked family endpoints.

**Evidence role of Phase 2:** design data. P2.1–P2.5r-LLM runs are never pooled into Phase 4 analysis.

**Not this protocol:** Q1 Holm families; OpenELM / LLMatic / in-context QD reproduction; Zelda as a second public family; Sphere or dungeon as public tasks; maze `solvability_repair`.

---

## 1. Selected tasks (two families)

| ID | Family | Task | Evaluator | Search metric | BC (frozen) | Repair |
|---|---|---|---|---|---|---|
| `nas201` | NAS-Bench-201 | TSS 15 625 cells | NATS-Bench TSS **simple** lookup | `cifar10-valid` accuracy (%), `hp="200"`, `is_random=False` | P2.1 20×20 `log10(params)` × `log10(flops)` | `identity` + genetic fallback |
| `pcg_sokoban` | PCG Benchmark | `sokoban-v0` 5×5 | `pcg_benchmark==0.1.0` @ `cd0f55b26c412a26e8797193e5417f5e651cf6cd` | scalar `quality` ∈ [0,1] | P2.5r `sokoban_v0_bin_edges_structural_counts.json` (solution_length 0…34 × crates 1…3, 10×10) | **`structural_counts` matched on every arm** |

Pins (immutable unless dated amendment):

- NAS-Bench-201 repo `7fa5a61464e454e96bdb90818f0d795d68d5728b`; NATS-Bench `1d4a304ad1906aa5866563438fcbf0d624b7eda2`; compact JSONL sha256 `6a0d9d1acd6aa939991af757082fdf188156b065fb96c267bbe70ca7f1b5f411`; official pth sha256 `2317a7dc911654322f4f932730a821cf38ecc5b5f1adde0aff26ff6b3f736666` **must not** be `torch.load` in-process next to the IDE.
- Compact table contains **no test keys**. Hold-out `cifar10` `x-test` is a post-search readout function only. Using test accuracy in fitness, selector, replacement, or prompts is a protocol violation (DROP the run, do not Holm it).
- PCG identity bin edges (`sokoban_v0_bin_edges.json`, `solution_length` ≡ 0) are **not** confirmatory BC.
- Zelda is one family with sokoban, not a second public task. It is not run in Phase 4.

Historical LifeManifold CA and maze remain motivating / Q1 evidence. They are not Holm families in this protocol and are not pooled with `nas201` / `pcg_sokoban`.

---

## 2. Unit of analysis and pairing

Primary unit: one independently seeded MAP-Elites run from a frozen initial archive.

Pairing: `pair_id = seed`. Arms in the same Block × domain that share `seed`, `initial_archive_hash`, model pin, and API-block identity are paired. Equal seeds without equal initial archives do not pair.

Candidate events inside a run are not replicates.

---

## 3. Replication, horizon, initialization

| Quantity | Value |
|---|---|
| n paired seeds (primary) | **20** (`301001`–`301020`) |
| overflow seeds (abort replacement) | `301021`–`301040`, used in order |
| initialization | `generated_floor`, **20** random structurally valid proposals, **no LLM**, generator `random`, then the search horizon |
| search horizon H | **200** proposal slots after the floor |
| total evaluations per run | 220 |
| initial-archive generator | NAS: `random_spec`; PCG: `random_spec` then `structural_counts` |
| floor RNG | `numpy` default_rng(seed) |
| occupancy floors for Block D | see §8; independent of LLM arms |

Confirmatory arms MUST NOT start from an archive produced by a tested LLM arm.

---

## 4. Model, decoding, prompts

**Primary model (all complete factorial cells):** request `gpt-4o-mini`; accepted response snapshot `gpt-4o-mini-2024-07-18`. If the API returns a different `response_model`, stop that family and dated-amend before continuing. Temperature **0.2**, `top_p` provider default (`null`), `max_tokens` **350**, retries **2** with existing backoff. Feasibility used this snapshot for GO batches; those batches stay design data.

**Robustness subset (not Holm):** Block A `nas201` only, n=10 seeds `303001`–`303010`, request `gpt-4o`, record `response_model`. Role: `robustness`. If a 20-proposal `design_pilot` parse rate is &lt; 0.50, drop the robustness subset and report it; do not change primary families.

**Constant / ablated prompt channel:** committed feasibility templates, byte-unchanged:

- NAS system/user: `prompts/nas201_llm_emitter_system.txt`, `prompts/nas201_llm_emitter_user.txt` (P2.2 SHA-256 `2c0b53f5a3c43b11db18af6bc4a4b5d8ca927b9d7605c4680c417d2c12fd358d`).
- PCG system/user: `prompts/pcg_sokoban_llm_emitter_system.txt`, `prompts/pcg_sokoban_llm_emitter_user.txt` (P2.4/P2.5r SHA-256 `10836e2c62db3e5f569950d1c57677ed4f84308a6c2de87acdeca4b8361b746f`).

**Live prompt channel (Block A only):** same **system** templates. Live **user** templates are new files written in P3.1 and hashed **before stamp**. Allowed fields only:

- NAS live user: parent ops JSON; parent `cifar10-valid` accuracy (%); parent params; parent flops. Forbidden: test accuracy, coverage, QD, archive, niche, few-shot cells, other datasets.
- PCG live user: parent grid JSON; parent `quality`; parent `solution_length`; parent `crates`. Forbidden: coverage, QD, archive, niche, Microban/README grids, few-shot language.

Few-shot count is 0. Copy-rate vs README example remains an audit flag, not a skip.

Prompt scan: constant templates keep the feasibility forbids (including the word `fitness` in NAS/PCG constant files). Live user templates have a separate scanner that allows the listed parent metrics and still forbids test / QD / coverage / archive / few-shot. A failed scan aborts the batch; do not read it as GO.

---

## 5. Treatment components (defaults unless a cell differs)

| Component | `nas201` | `pcg_sokoban` |
|---|---|---|
| initialization | generated_floor 20 | generated_floor 20 |
| selector | cell-specific | cell-specific |
| generator | cell-specific | cell-specific |
| prompt_channel | `constant` or `live` | `constant` or `live` |
| repair_fallback | `identity` + `genetic_fallback` (one edge) | **`structural_counts` after every emit**, including fallback; fallback is one-tile genetic then repair |
| gate | `off` | `off` |
| replacement | strict single elite, fitness maximize | same, fitness = quality |
| allocation | `static` unless Block C | same |
| budget | H=200 proposals after floor; LLM arms: ≤1 completed LLM call per proposal slot under `static` | same |

Arm labels such as `LLM` or `hints` are invalid without this vector. Manifest hash is identity.

Genetic primary operator: NAS one different op on one edge; PCG one different tile. This is the smoke operator, **not** calibrated to isolated LLM Hamming. A k-step genetic match to mean Hamming is **not** a confirmatory arm in this freeze (may be a later Extension with a new family ID).

---

## 6. Blocks and cells

Shared across a domain unless noted: model pin, floor, H, repair, gate, replacement.

### 6.1 Block A — prompt channel × selector (required both tasks)

Four LLM cells:

| arm_id | prompt_channel | selector |
|---|---|---|
| `A-const-uniform` | constant | `uniform_frontier` |
| `A-const-minfit` | constant | `min_fitness_frontier` |
| `A-live-uniform` | live | `uniform_frontier` |
| `A-live-minfit` | live | `min_fitness_frontier` |

Generator = LLM. Allocation = static.

### 6.2 Block B — generator (required both tasks)

Selector `uniform_frontier`, prompt `constant` (N/A for non-LLM), allocation static.

| arm_id | generator | Notes |
|---|---|---|
| `B-llm` | LLM | **same treatment as `A-const-uniform`** — do not rerun; reuse those 20 runs |
| `B-genetic` | one-step genetic | |
| `B-random` | random valid | PCG: random then `structural_counts` |

### 6.3 Block C — LLM-call allocation (both public tasks)

Generator LLM, prompt `constant`, selector `uniform_frontier`. Fixed **LLM-call cap = 200** completed calls and **proposal horizon = 200** after floor.

| arm_id | allocation |
|---|---|
| `C-uniform` | `static` — **reuse `A-const-uniform`** |
| `C-early` | spend the 200 LLM calls on the first 200 search slots (identical to static here if every slot is LLM). **If static already is one LLM per slot, `C-early` is identical to `C-uniform` and is not run.** |
| `C-state` | `state_aware_median` (§6.4) |

Because Block A static LLM already uses one call per slot, **early vs uniform is not identified** under this H and cap. It is **not** a Holm contrast. Block C Holm is only `C-state` vs `C-uniform` (m=1 per domain). Report forfeited LLM calls under `C-state`.

If a later amendment lowers the LLM-call cap below H, `C-early` becomes distinct and requires a new family ID; do not silently add it to F-C.

### 6.4 Frozen state-aware rule

At each search slot:

1. Select the target cell with `uniform_frontier` (empty-archive fallback: uniform among all cells).
2. Let `m` be the median fitness of currently occupied elites, or `−∞` if the archive is empty.
3. Call the LLM iff the target cell is empty **or** its elite fitness ≤ `m`.
4. Otherwise emit one-step genetic (then PCG `structural_counts`).
5. If completed LLM calls have already reached 200, remaining slots are genetic.
6. Forfeited calls = `200 − completed LLM calls`. They count as unused budget, not as genetic proposals.

This rule is not tuned on Phase 2 batches.

### 6.5 Block D — occupancy (NAS only, descriptive)

n=10 seeds `304001`–`304010`. Treatment = `A-const-uniform`. Two independently generated floors: 20 random (same as primary) and 50 one-step genetic (no LLM). Role: `sensitivity`. Not Holm. PCG Block D is not in this freeze (A* cost; occupancy not assumed comparable).

---

## 7. Confirmatory families and multiplicity

α = 0.05 two-sided. Test: paired Wilcoxon signed-rank on n=20 seed-level differences of the **primary endpoint**. Holm correction **within the named family only**. No paper-wide Holm. No pooling NAS with PCG. Interaction in the Block A 2×2 is **pre-declared descriptive** (report the contrast; do not Holm it).

**Primary endpoint (all Holm families):** anytime AUC of **QD-score**, proposal-indexed. Drop the first `floor_random` points of the series; the random floor is not an intercept and is not inside the integral. On the remaining n points (n = H = 200 when the search completes), the AUC is the equal-spaced trapezoidal area divided by (n − 1). An empty series is 0. A single point is that point. QD-score = sum of elite fitness on occupied cells of the frozen grid.

**Locked companion (not Holm):** terminal QD-score at slot H. If companion and primary disagree in sign, the locked family still follows anytime AUC; the companion is dual-reported, not a rewrite.

**Explicitly not Holm:** NAS coverage (structural ceiling 21/400 bins); PCG coverage; playable fraction; parse/fallback rates; token/cost; NAS post-search test accuracy; Block A interaction; robustness `gpt-4o`; Block D.

| Family | Domain | Contrasts (Holm) | m |
|---|---|---|---|
| `F-A-nas` | nas201 | (1) minfit − uniform \| constant; (2) live − constant \| uniform | 2 |
| `F-A-pcg` | pcg_sokoban | same two | 2 |
| `F-B-nas` | nas201 | (1) LLM − genetic; (2) LLM − random | 2 |
| `F-B-pcg` | pcg_sokoban | same two | 2 |
| `F-C-nas` | nas201 | (1) state_aware_median − static | 1 |
| `F-C-pcg` | pcg_sokoban | (1) state_aware_median − static | 1 |

Family **pass** = all Holm-adjusted p-values in that family &lt; 0.05. Family fail does not delete the family ID.

No SESOI / TOST in this freeze. Do not add equivalence claims after seeing CIs.

---

## 8. Budget axes, cap, price table

Report proposal-, evaluation-, and LLM-call-indexed anytime views in the analysis. Token, wall-time, and monetary views are required in the artifact bundle even if compact in the paper.

**Cap (stop launching new confirmatory LLM runs if crossed):** 90 000 completed LLM calls **or** 40 × 10^6 total tokens **or** 14 calendar days after the first confirmatory API call, whichever first. An incomplete family is reported as incomplete; do not compute Holm on a partial seed vector unless a dated amendment froze a reduced n **before** inspection.

**Monetary:** USD. Dated price table **attached for stamp** at
`artifacts/controlled_attribution/price_tables/2026-09-02/`:

| Artifact | SHA-256 |
|---|---|
| `price_table.json` (`openai-public-list-2026-09-02`) | `a60d1f99b89e7067b8fb2e1eac0c9a7f2b64c041f460101e864c2b413e549b94` |
| `openai_gpt-4o-mini.md` snapshot | `6685868cf5e31ed39492380b2e3b8b7b430437971ed0c3d742319d54f8284d78` |
| `openai_gpt-4o.md` snapshot | `a0bd691d562c3af1dcf8c8e795ea400c05f53a8ca1feccaa632f803a8dbcdd6d` |
| `openai_pricing.md` index snapshot | `14ad7ca4328a97587487626a637a60b59235632d6b8d6cbb864bf371120367ae` |

List rates (Standard, USD / 1M tokens, retrieved 2026-09-02 from OpenAI model pages):

| Request model | Role | Input | Cached input | Output |
|---|---|---|---|---|
| `gpt-4o-mini` (pin `gpt-4o-mini-2024-07-18`) | primary | 0.15 | 0.075 | 0.60 |
| `gpt-4o` | robustness (not Holm) | 2.50 | 1.25 | 10.00 |

Monetary reporting uses these rates on logged token splits. Hard stop remains calls / tokens / days. Upper bound if all 40M cap tokens were billed as mini **output**: **USD 24**. Do not treat that bound as a second hard stop.

Illustrative order (not a second freeze): Block A 2×4×20×200 = 32 000 calls; Block B extra 2×2×20×200 = 16 000; Block C extra 2×1×20×200 = 8 000; robustness 4×10×200 = 8 000; total ≈ 64 000 &lt; 90 000.

---

## 9. Quality gates and failure handling

Per proposal: structurally invalid → no NAS lookup / no PCG A* (P8/N3). Parse fail → genetic fallback (then PCG repair). Fallback consumes the proposal slot and the LLM-call attempt.

Per run (primary analysis is ITT): keep the run if 200 search slots completed. Log parse rate, fallback rate, repair Hamming (PCG). Sensitivity (pre-declared, not Holm): drop pairs where any paired arm has parse rate &lt; 0.50.

Per pair abort (process crash, persistent API failure after retries): exclude the **whole pair** (all arms of that block×domain×seed) and replace with the next overflow seed. Do not Holm a mix of complete and patched arms inside one pair.

Do not stop a family because an interim contrast looks small.

---

## 10. Execution, blocking, inspection log

Order: `F-A-nas` runs complete on disk → freeze those run manifests → **then** first inspection of `F-A-nas` → then `F-A-pcg` → `F-B-*` (reuse A where specified) → `F-C-*` → robustness → Block D.

Calendar/API block: record UTC date of each call in the existing LLM call log. Interleave arms within a pair (A-const-uniform seed k, A-const-minfit seed k, …) to avoid one arm owning a whole outage day.

**First inspection:** a line in `artifacts/controlled_attribution/PHASE4_INSPECTION_LOG.md` (created at first look) with UTC datetime, family ID, and operator initials. If the log is missing, do not invent it (Q1 gap).

In-process lookup and PCG `quality` are allowed. Do not `torch.load` the official NAS pth. Do not pickle-load bundled NATS `.pickle.pbz2` in the IDE process.

---

## 11. Release and privacy

Public: treatment/run manifests, genotype hashes, compact metrics, sanitized call-log metadata (model, tokens, latency, prompt SHA). Private: raw provider payloads if they can contain secrets; API keys never in git.

NAS compact table remains without test metrics. Generated PCG grids are releasable (MIT). Prompt text is releasable.

---

## 12. Exclusions from this freeze

- Zelda LLM or zelda confirmatory cells.
- `sokoban-complex` / `*-large`.
- Identity-repair PCG as a Holm arm.
- Maze/dungeon/sphere public-task substitution.
- Surrogate gate or prompt hints from the CA MLP.
- Few-shot Microban / NAS examples.
- Paper-wide Holm, TOST, SESOI.
- Optional §11 headline reproductions.
- Changing NAS BC to op-histogram.
- Moving G5 Jaccard threshold.

---

## 13. What must exist before stamp (P3.1)

1. `ReadOnlyNormalizationAdapter` + native loop for `nas201` and `pcg_sokoban` (sidecar, not a new universal QD kernel). **Done** — see `PHASE3_P3_1.md`.
2. Live user prompt files + scanners + SHA-256 inserted into this protocol by dated amendment **still pre-data**. **Done** — §13.1.
3. Capabilities: `allocation_kinds` include `static` and `state_aware_median`; PCG `repair_fallback_kinds` include `structural_counts`. **Done**.
4. Job builder emits the cells in §6 without launching. **Done** — `scripts/build_confirmatory_jobs.py`.
5. Admission tests: refuse feasibility JSONL paths, refuse Q1 nightly summaries, refuse mixed `evidence_tier`. **Done** — `worldspace/attribution/confirmatory_admission.py` + tests.
6. Q1 scheduler YAML and archived `nightly_run_summary.json` byte-unchanged.

Stamp after those hashes are in the protocol text. Do not stamp an incomplete prompt list.

### 13.1 Pre-data prompt SHA-256 (2026-09-02 amendment)

| Path | SHA-256 |
|---|---|
| `prompts/nas201_llm_emitter_system.txt` | `373ecefdfb040e7b8a5cbc79ff400522cc509512c68f071209522c066cbe3b00` |
| `prompts/nas201_llm_emitter_user.txt` | `95e192f17552450d616bde1ff7e95fc1bf1b76dcb204d563d4ab56cd7f0ae1e5` |
| `prompts/nas201_llm_emitter_live_user.txt` | `a96af227263d9ae41fb49d903c6443590414b626588c8d658a43a865bc0d4233` |
| `prompts/pcg_sokoban_llm_emitter_system.txt` | `50baddc3d9cb159fded244938dce2253e83818b5c6b4fd9e90b204a2e6b8e687` |
| `prompts/pcg_sokoban_llm_emitter_user.txt` | `874cf140f29192b8ee193eb6922738c4ca7e623f9455ba5754f14d0b401b7584` |
| `prompts/pcg_sokoban_llm_emitter_live_user.txt` | `caf5778b9390aa334437bdbb595221d159d0616fbaa616df800a8b243edda694` |

Live channel: parent metrics only (`parent_*` placeholders). Scanners refuse test split, archive/population QD, few-shot. Constant channel templates remain the Phase 2 prompt files above.

### 13.2 P3.2 stamp procedure (external, then tag)

Still **pre-data** until both exist:

1. Register this file's **byte SHA-256** (see `PHASE3_P3_2.md` / `phase3_protocol_draft.json` after hashing) on OSF Registrations or AsPredicted.
2. Commit that carries this byte-identical `PHASE3_PROTOCOL.md`, then git tag `phase3-protocol-v0` on that commit.
3. One Reporting line records the registry ID pointing at the same SHA.

Do **not** set `freeze_in_force: true` or create the tag from a chat alone. Do not start Phase 4 LLM until both are true.

### 13.3 Pre-data AUC formula (2026-11-01 amendment)

Section 7 is the only QD-score anytime AUC: drop the floor, then trapezoidal area / (n − 1). The earlier wording "divided by H" is withdrawn, still pre-data. This amendment does not freeze the protocol and does not authorise Phase 4.

Price table: **attached** (§8). Prospective event capture on public runners remains optional.

---

## 14. Locked claim sentences (freeze at external stamp)

Exact sentences locked when the registry + tag pair in §13.2 exists:

1. **Bundling / interaction (descriptive, not Holm):** Under matched repair and budget, Block A tests whether the selector effect and the live-vs-constant channel effect add as a simple interaction on QD-score anytime AUC; the interaction contrast is reported descriptively and is not a Holm family endpoint.
2. **Channel (Holm F-A):** Under constant grammar elsewhere matched, live parent-metric prompts change QD-score anytime AUC relative to constant prompts at uniform frontier selection.
3. **Selector (Holm F-A):** Under the constant prompt channel, min-fitness frontier selection changes QD-score anytime AUC relative to uniform frontier selection.
4. **Generator (Holm F-B):** Under matched selector, repair, and allocation, the LLM generator changes QD-score anytime AUC relative to one-step genetic and relative to random.
5. **Allocation (Holm F-C):** Under matched constant channel and uniform selector, state-aware median LLM allocation changes QD-score anytime AUC relative to static one-LLM-call-per-slot allocation.
6. **Domain heterogeneity (descriptive):** Family pass/fail for F-A/B/C may differ between `nas201` and `pcg_sokoban`; those outcomes are compared without a joint cross-domain Holm vector.
7. **Non-claims:** This protocol does not claim universal LLM superiority, portable percentage-point effects across domains, causal effects of omitted stack components, or any use of NAS test-split metrics in search.

---

## 15. Amendment rule

After freeze: new cells or endpoints need a new family ID (Extension). Outcome write-ups are Reporting. Category corrections that change what a locked family *means* require dual-report of the locked result plus the correction (Q1 H4 lesson). Do not demote a Holm family to descriptive because it failed.
