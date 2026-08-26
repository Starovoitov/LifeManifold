# RQ1 second-domain attribution — maze, done properly

**Status:** Phase A complete 2026-08-17 (10/10). Phase B complete 2026-08-18 (**40/40**).
Primary @ 5k: minfit **16.79 ± 3.29%** vs frozen uniform **56.98 ± 4.47%**;
Δ **−40.19 ± 4.13 pp** (10/10). Readout: [`experiments/q1-v5-maze-genetic-minfit/ANALYSIS.md`](experiments/q1-v5-maze-genetic-minfit/ANALYSIS.md).
Phase B LLM 2×2: policy **+39.10 / +39.31 pp** (10/10); leftover **+0.02 / +0.23 pp**.
Readout: [`experiments/q1-rq1-maze-factorial/ANALYSIS.md`](experiments/q1-rq1-maze-factorial/ANALYSIS.md).
**Does not amend** Holm H5, locked maze arms, CA RQ1, or genetic H2.

Maze H5 is a hard-gate transfer family. This protocol is a **policy × soft**
attribution replay on a second evaluator, not a reread of H5.

## Why maze (and why not H5-as-is)

Maze is a different genotype (16×16 tiles), different evaluator (BFS path +
structure), and already has an LLM emitter. Frozen genetic `uniform_frontier`
at 32.5k saturates the 900-cell archive (**~99.6%**), which is why H5 uses
AUC. Frozen maze hints−stub is **+1.5 pp AUC**, not +15 — there is nothing
to collapse at that ceiling.

Fitness–descriptor coupling exists but is **not** the CA \(w_D\) story:
\(f = 0.55\,\mathrm{length} + 0.30\,\mathrm{branch} + 0.15\,\mathrm{reach}\);
BC = \((\mathrm{path\_measure}, \mathrm{branching\_measure})\). On frozen
genetic seed 0, Spearman \(\rho(f,\mathrm{path}){\approx}0.89\),
\(\rho(f,\mathrm{branch}){\approx}0.42\). Operators preserve solvability
(no \(f{=}0\) unsolvable dump).

Dungeon is the same PCG-BFS family. Sphere H2 has no LLM; the Sphere H1
2×2 is a separate protocol: [`Q1_RQ1_SPHERE_DOMAIN.md`](Q1_RQ1_SPHERE_DOMAIN.md).
Neither is this maze Phase A.

## Phase A — genetic policy probe (cheap, required)

**Question:** on maze, with archive headroom, does `min_fitness_frontier`
vs `uniform_frontier` move coverage the way it does on CA?

| Knob | Frozen `genetic` (H5) | This arm |
|------|----------------------|----------|
| Emitters | 20R+30G | 20R+30G |
| Policy | `uniform_frontier` (hardcoded historically) | `min_fitness_frontier` |
| LLM / surrogate | off | off |
| Archive | 30×30 / 900 | same |
| Budget | 32.5k (do not re-run) | 32.5k, new tier |

**Tier:** `q1-v5-maze-genetic-minfit`
**Scheduler:** `worldspace/specs/maze_scheduler_genetic_minfit.yaml`
**Comparator:** frozen `artifacts/experiments/q1-v5-maze-full/genetic/` traces
(do **not** re-run uniform).

**Primary endpoint (locked before launch):** coverage at **5{,}000**
evaluations. Frozen uniform there is **57.0 ± 4.5%** (range 51.7–65.3;
n=10). That is mid-fill headroom; 15k is already ~97%.

**Secondary:** coverage @ 8k / 10k; terminal 32.5k (likely ceiling, not the
claim); unique-target concentration if logged.

### Go / no-go (descriptive; not Holm)

Read paired seeds 0–9, minfit minus frozen uniform, at 5k evaluations:

| Decision | Rule |
|----------|------|
| **GO** Phase B | \(\lvert\mathrm{mean}\,\Delta\rvert \ge 5\) pp, **or** 8/10 same sign and \(\lvert\mathrm{mean}\,\Delta\rvert \ge 3\) pp |
| **NO-GO** | \(\lvert\mathrm{mean}\,\Delta\rvert < 2\) pp at **both** 5k and 8k |
| **Borderline** | otherwise: report, do **not** spend LLM; optional maxfit / larger archive, not Phase B |

NO-GO means maze policy does not move coverage in the headroom regime, so a
stub/hints 2×2 cannot replay the CA inversion. That is a valid negative
(CA-geometry magnitude), not a failed attribution paper.

### Launch

```bash
./scripts/run_experiment_batch.sh q1-v5-maze-genetic-minfit 0 9
```

Log: `artifacts/experiments/q1-v5-maze-genetic-minfit/logs/batch.log`

CPU only; maze genetic is ~25–32 s/seed (~5 min serial). Full 32.5k is for the anytime curve
against frozen traces; the decision cut stays 5k.

### Result (2026-08-17)

**GO.** @ 5k: minfit **16.79 ± 3.29%** vs uniform **56.98 ± 4.47%**;
paired Δ **−40.19 ± 4.13 pp** (10/10). @ 8k −53.2 pp; terminal
45.11 ± 5.39% vs 99.62 ± 1.19% (−54.5 pp). Uniform saturates; minfit
does not. Details: [`ANALYSIS.md`](experiments/q1-v5-maze-genetic-minfit/ANALYSIS.md).

## Phase B — LLM 2×2 (only on GO)

Four cells, **not** five floors, **not** Holm, **not** H5:

`{stub, live} × {minfit, uniform}` on maze, dated `gpt-4o-mini-2024-07-18`,
call logs.

### Lock (2026-08-17, after Phase A GO)

| Knob | Lock |
|------|------|
| Archive | **Empty** (not occupancy-matched to CA 38.84%; not a warm-start continuation) |
| Budget | **5{,}000** proposal slots (100 × 50); primary endpoint = terminal coverage |
| Model | `gpt-4o-mini-2024-07-18` via `worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml` |
| Call logs | on (`run_maze_qd.py` → `llm_call_log.jsonl`) |
| Cells | `llm_stub_minfit`, `llm_stub_uniform`, `llm_hints_minfit`, `llm_hints_uniform` |
| Seeds | **0–9** (n=10; 40 runs) |
| Surrogate | off on stub; `maze_v1.pkl` on live (same checkpoint as H5 hints; gate off) |
| Emitters | 20R+30L (implicit `llm_*` mix) |
| Tier | `q1-rq1-maze-factorial` → `artifacts/experiments/q1-rq1-maze-factorial/` |
| Locked H5 files | **do not edit** `maze_scheduler_llm_stub.yaml` / `llm_hints.yaml` (650 iters) |

**Regime:** maze headroom where genetic policy Δ is **−40.19 pp** @ 5k. This is
not CA mid/late continuation. Disclose as a second-evaluator empty-start 2×2.

**Contrasts (descriptive):** policy @ stub and @ live; leftover @ minfit and
@ uniform. Do **not** treat locked H5 `llm_stub` / `llm_hints` (uniform, 32.5k,
ceiling) as two cells of this grid.

**Launch:**

```bash
./scripts/run_rq1_maze_factorial_nohup.sh
# override: MAZE_FACTORIAL_WORKERS=4 ./scripts/run_rq1_maze_factorial_nohup.sh
```

Progress: `python scripts/analyze_rq1_maze_factorial.py`

**Reporting:** Descriptive. Not a new Holm family. Does not identify CA Holm
+15.09. Manuscript: maze 2×2 bound, not a new primary RQ.

### Result (2026-08-18)

**40/40.** Terminal coverage @ 5k (empty): stub minfit **5.10 ± 0.83%** /
stub uniform **44.20 ± 3.52%** / hints minfit **5.12 ± 0.81%** / hints
uniform **44.43 ± 3.68%**. Policy **+39.10 ± 3.38** / **+39.31 ± 3.97 pp**
(10/10 both soft levels). Leftover **+0.02 ± 0.14** @ minfit (5/2/3);
**+0.23 ± 3.42** @ uniform (5/4/1). Same inversion as CA (policy dominates;
scalars do not). Magnitudes are maze empty-start @ 5k, not CA continuation
pp. Details: [`ANALYSIS.md`](experiments/q1-rq1-maze-factorial/ANALYSIS.md).
