# H1 stronger soft actuator: child-level rewrite (Path A)

**Status:** designed; **pilot not launched**  
**Tier:** `q1-h1-child-rewrite-pilot` → `artifacts/experiments/q1-h1-child-rewrite-pilot/`  
**Stack:** LLM mixed emitters (`20R+20G+10L`), `uniform_frontier`, warm-start baseline, checkpoint `nightly_v3_mc_d005`, primary `qwen-turbo`  
**Why:** confirmatory H1 only swaps parent-level surrogate scalars that restate archive-true fitness. Path A tests a **non-redundant** soft channel: generate a draft child → `predict(child)` → optional second LLM rewrite using that child-level prediction.

## Claim scope (after readout)

Admissible if positive:

> Under the LLM mixed stack, a child-level rewrite pass guided by surrogate predictions on the **draft child** changes terminal / eval-indexed coverage relative to single-pass `hints`.

Admissible if null:

> Adding a child-level rewrite pass (triggered when predicted child fitness is below parent true fitness) did not move coverage beyond single-pass parent-scalar hints; stronger soft value remains unproven for this actuator.

Not claimed either way: general before-generation impossibility; equivalence to hard gating (H2).

## Mechanism

Per LLM slot (`≈10/50` batch):

1. **Pass 1 (draft)** — identical to confirmatory `hints`: parent elite JSON + live parent surrogate scalars → LLM → draft `WorldSpec`.
2. **Surrogate on child** — `SurrogateFacade.predict(draft_child)` (not parent).
3. **Trigger** (default `below_parent_true`) — rewrite iff  
   `predicted_child_fitness < parent_true_fitness`  
   (parent true fitness from archive elite; empty niche → always rewrite if draft parsed).
4. **Pass 2 (rewrite)** — second LLM call with rewrite user prompt containing:
   - target niche,
   - parent true fitness,
   - draft child JSON,
   - predicted child fitness / uncertainty,
   - instruction to edit the draft so predicted fitness improves while staying in niche.
5. **Commit** — use rewritten child if parse succeeds; else keep draft (`keep_draft_on_rewrite_fail: true`).

Fallback (`llm_fallback` random-walk) never enters rewrite.

## Arms

| Condition | Scheduler | Role |
|-----------|-----------|------|
| `hints` | `map_elites_scheduler_nightly_llm.yaml` | Single-pass baseline (reuse `q1-v3-mixed-2x2/hints` if pin-equal) |
| `hints_rewrite` | `map_elites_scheduler_nightly_llm_hints_rewrite.yaml` | Pass 1 + child predict + conditional rewrite |

Held fixed vs confirmatory H1: emitters, warm-start, seeds, checkpoint, budget 650×50, `archive_trace`.

**Not** in this tier: shuffled placebo, cold-start (separate packages).

## Config (locked)

```yaml
llm:
  enabled: true
  child_rewrite:
    enabled: true
    trigger: below_parent_true   # always | below_parent_true | below_tau | below_parent_pred
    min_predicted_fitness: 0.45  # used only when trigger=below_tau
    keep_draft_on_rewrite_fail: true
    user_prompt_path: prompts/map_elites_llm_emitter_user_rewrite.txt
```

Metadata: successful rewrite → `emitter_type=llm_rewrite`; draft kept → `llm`; parse fail → `llm_fallback`.

## Pilot protocol (stop/extend)

| Stage | Seeds | Gate |
|-------|------:|------|
| Pilot | 0–2 | Extend if mean \|Δcov\| (`hints_rewrite`−`hints`) ≥ **2 pp** terminal **or** @20k eval |
| Full | 0–9 | Only if pilot gate passes |

Pilot Δfit companion: extend also if mean Δfit ≥ +0.02 (same as prior soft pilots).  
If pilot fails both gates → **do not** full n=10; report as negative pilot (like `hints_direction`).

## Endpoints (descriptive)

- Terminal coverage / QD; cov@5k / @10k / @20k; AUC cov@20k
- Rewrite rate: triggered / accepted / kept-draft
- Accepted-elite LLM quality split by `emitter_type` (`llm` vs `llm_rewrite`)
- Pred→realized: among rewritten slots, correlation of Δpred (rewrite−draft) with Δtrue when both evaluated (accepted only unless rejected logging lands)

## Cost

- `hints`: reuse preferred (~0 new calls if pin match)
- `hints_rewrite`: ≤2× LLM calls on LLM slots → worst case ~13k calls/seed; with `below_parent_true` expect much less than 2× (log rewrite rate)
- Pilot 3 seeds ≈ 1–2 days wall at 2 workers

## Launch

```bash
# Pilot (default seeds 0–2; tier default LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2)
./scripts/run_experiment_batch.sh q1-h1-child-rewrite-pilot

# Or explicit
LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2 ./scripts/run_experiment_batch.sh q1-h1-child-rewrite-pilot 0 2

# Analyze
.venv/bin/python scripts/analyze_h1_child_rewrite.py
```

Requires `QWEN_API_KEY` (source `.env`). Prefer `MIXED_2X2_WORKERS`-style load discipline if relaunching nohup helpers later.

## Preflight

```bash
uv run python scripts/preflight_h1_child_rewrite.py
```

GO if: checkpoint loads; draft→predict(child) returns finite fitness; rewrite prompt formats; trigger `below_parent_true` fires on synthetic pred_child < parent_true.

## Analysis artifacts

```
artifacts/experiments/q1-h1-child-rewrite-pilot/
  hints/                  # or symlink / reuse note
  hints_rewrite/
  summary.csv
  ANALYSIS.md
  h1_child_rewrite_analysis.json
```

Script: `scripts/analyze_h1_child_rewrite.py`.

## Relation to failed soft pilots

| Pilot | Verdict |
|-------|---------|
| `hints_rich` / `hints_parent` / `hints_direction` | null @ n=1 — parent-side / FD advice |
| **child rewrite (this)** | first confirmatory-style test of **child-level** soft advice |

Do not pool with H2 filter; rewrite still evaluates every committed child (no skip).

## Manuscript

Wire after pilot/full readout into `\S\ref{sec:h1-impl-threat}` / Future work. Until then: designed, not claimed.
