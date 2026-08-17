# RQ1 second-domain attribution — maze, done properly

**Status:** Phase A complete 2026-08-17 (10/10). **GO** for Phase B.
Primary @ 5k: minfit **16.79 ± 3.29%** vs frozen uniform **56.98 ± 4.47%**;
Δ **−40.19 ± 4.13 pp** (10/10). Readout: [`experiments/q1-v5-maze-genetic-minfit/ANALYSIS.md`](experiments/q1-v5-maze-genetic-minfit/ANALYSIS.md).
Phase B (LLM 2×2) **not launched**.
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

Dungeon is the same PCG-BFS family. Sphere has no LLM. Neither is Phase A.

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

**Budget:** 5k proposals/arm (same primary cut), not 32.5k. Maze 30L would
otherwise be ~19.5k LLM calls/seed.

**Warm-start:** empty or a shared mid-fill floor truncated near 40% — lock
in a Phase B amendment after GO, not now.

**Do not** treat locked H5 `llm_stub` / `llm_hints` (uniform, 32.5k, ceiling)
as two cells of this grid.

## Reporting

Descriptive. Not a new Holm family. Does not identify CA Holm +15.09.
Manuscript stays one-evaluator until Phase B exists **or** Phase A is
NO-GO and is written as a bound.
