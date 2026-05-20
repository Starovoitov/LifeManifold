# WorldSpace MVP Architecture

This document describes the new "world space" architecture for the project.

## Goal

Represent each cellular automata world as a point in parameter space, simulate behavior, extract behavior metrics, and map worlds into a searchable/comparable embedding.

Pipeline:

`Generator -> WorldSpec JSON -> Simulator -> Metrics -> Embedding/Clustering -> World Space`

## Core Concepts

### WorldSpec

File: `worldspace/specs/spec.py`

`WorldSpec` is the canonical world description (JSON-serializable):

- `birth`: list[int]
- `survival`: list[int]
- `noise`: float
- `resource_regen`: float
- `predation`: float
- `cell_types`: list[str]
- `neighborhood`: str (default `moore`)
- `grid_size`: int
- `steps`: int
- `seed`: int

Each world is a point in rule space and can be saved/loaded as JSON.

### Generator Ladder

File: `worldspace/generators/__init__.py` (package `worldspace.generators`)

Implemented generator levels:

1. `random_walk(...)` utility (small local parameter changes)
2. `RandomWorldGenerator` (independent random worlds)
3. `RandomWalkWorldGenerator` (trajectory in world space)
4. Markov hierarchy:
   - `MarkovWorldGenerator` (base class)
   - `TwoStateNoiseMarkovGenerator` (calm/chaos switching)
   - `RuleBiasMarkovGenerator` (survival/reproduction bias)
5. `GeneticWorldGenerator` (PyGAD-backed evolution on `mo_eoc_indicator`)
   - chromosome encoding: rule masks + scalar params
   - YAML-driven GA knobs via `genetic_world_generator.yaml`
6. `LLMWorldGenerator` (iterative LLM-guided local search)
   - loop: simulate → score → ask LLM for slight patch → validate/clamp → next world
   - local/remote provider routing via `llm_world_generator.yaml`
   - robust fallback to random-walk mutation if LLM response is invalid
7. `HybridGALlmWorldGenerator` (population evolution with mixed mutations)
   - selection: top-k plus random diversity sample
   - offspring mix: random GA mutation + LLM-guided mutation
   - LLM branch sees top fraction of selected worlds (`llm_top_fraction`)
   - configured via `hybrid_world_generator.yaml`
8. `NeuralWorldGenerator` (YAML-driven latent MLP policy)

### Simulator

File: `worldspace/simulator.py`

`run_world(world, *, ca_step_trace_file=None, ca_step_trace_yield_index=0, early_extinction_step=None) -> SimulationResult`

Optional keyword-only args:

- `ca_step_trace_file` (pipeline / `--ca-step-trace`): after each CA timestep, append one JSON line (`yield_index`, `ca_step`, `metrics`).
- `early_extinction_step` (MAP-Elites illuminator, default `200`): stop when `life.mean() == 0` at post-init timestep `t` with `0 <= t < early_extinction_step`. `None` = legacy full `world.steps` run with no early stop.

Simulation model:

- 2D grid
- Moore neighborhood
- Birth/survival update
- Noise flips
- Predation deaths
- Resource regeneration and feed bonus

Returns `SimulationResult` with **`metrics: WorldMetrics`**, optional **`final_life`** / **`final_food`** for plotting and diagnostics, and **`early_extinct`** (set when `early_extinction_step` triggers a §2.1 stop).  
Inside the loop there are **no growing Python lists** of per-step grids or full density series: online statistics, a bounded density window for oscillation, and one final grid snapshot.

### MAP-Elites evaluation (illuminator core)

Package: `worldspace/illuminators/`

`evaluate_candidate(world_spec, *, resolution=50, early_extinction_step=200) -> EvalResult` wires canonical seed, `run_world`, behavioral `measures` (`stability`, `diversity`), `fitness`, and archive `bin`. Used by the illuminator loop before archive insert (see project MAP-Elites docs).

File: `worldspace/illuminators/archive.py`

`GridArchive(resolution=50)` holds at most one `ArchiveElite` per cell over BC range `[0, 1]`. `try_insert(elite) -> InsertResult` accepts any fitness into an empty cell, replaces only when `fitness_new > fitness_old` (strict), otherwise rejects without mutating the stored elite.

`insert_evaluated(archive, eval_result, metadata)` builds a full elite from `evaluate_candidate`. `insert_and_persist(..., jsonl_path)` appends one line to `output/map_elites_archive.jsonl` only when the insert is accepted (schema 1.2: `bin`, `world_spec`, `fitness`, `measures`, optional `metrics`, `metadata`).

`load_and_collapse_jsonl(path, resolution=50)` reads JSONL and keeps one elite per `bin` (maximum `fitness`; first line wins on ties). Invalid lines are skipped with a log warning by default (`on_invalid_line="raise"` for strict mode). `merge_archives(base, incoming)` applies the same strict-improvement rule across two in-memory archives. Typical resume flow: load collapsed archive, run new candidates with `insert_and_persist`, append only improvements to the file.

File: `worldspace/illuminators/grid_neighbors.py` — bounded cardinal/Moore neighbor coordinates for archive bins (no torus; distinct from toroidal Moore in `worldspace/math.py`).

File: `worldspace/illuminators/scheduler.py`

Production defaults live in `worldspace/specs/map_elites_scheduler.yaml` (`schema_version` 1.2, `batch_size` 50 with a fixed `batch_emitters` list: 20 random + 20 genetic + 10 llm, `initial_random_candidates` 100). CI reproducibility uses `worldspace/specs/map_elites_scheduler_mini.yaml` (`iterations` 20, `batch_size` 4, `grid_resolution` 10, `llm.enabled: false`) via `DEFAULT_MINI_SCHEDULER_PATH`; not the production CLI default. `load_scheduler(path, iterations_override=None) -> SchedulerConfig` validates the YAML (including `len(batch_emitters) == batch_size`). `resolve_emitter_for_slot(config, candidate_id, candidates_evaluated)` returns the effective emitter: while `candidates_evaluated < initial_random_candidates`, always `random` (ignoring the YAML slot); afterward the slot entry from `batch_emitters`. `RunCounters` holds `candidates_evaluated` across iterations; call `record_evaluation()` after each candidate run. `select_target_bin(archive, rng) -> TargetBin` picks a niche for the next candidate: uniform over the grid when the archive is empty, else occupied cells on the archive frontier (cardinal neighbor empty) with minimum elite `fitness` (lexicographic tie-break on `(i, j)`), else uniform random over the grid; returns bin indices plus BC centers via `bin_center`.

File: `worldspace/illuminators/illuminator.py`

`MapElitesIlluminator.run(...)` loads scheduler YAML, optional `load_archive_path` (collapsed JSONL), seeds one global `numpy.random.Generator` from CLI `--seed` (does not override per-candidate canonical hash seeds), builds `MapElitesEmitter`, writes only `{output_dir}/map_elites_archive.jsonl`, and runs `run_scheduler` for `iterations × batch_size` evaluations. Resuming a non-empty archive sets `candidates_evaluated` to `initial_random_candidates` so the initial random fill phase is not repeated.

CLI: `python -m worldspace --illuminator mapelites` via `worldspace/cli_mapelites.py` (`--seed`, `--grid-resolution`, `--iterations`, `--scheduler`, `--load-archive`, `--output-dir`, `--steps` ≥ 200, `--grid`). Legacy `--generator` path is unchanged.

CI smoke (E6.1): `tests/test_map_elites_smoke.py` runs the mini scheduler end-to-end via `MapElitesIlluminator`, writes persistent artifacts under `artifacts/map_elites_smoke/` (`map_elites_archive.jsonl`, `smoke_run_summary.json`), validates JSONL schema, and completes without LLM calls. GitHub Actions workflow `.github/workflows/map_elites_smoke.yml`; local: `make smoke-map-elites`.

Nightly: default pipeline in `worldspace/scripts/run_map_elites_nightly.py` (`make nightly-map-elites`): (1) baseline with `map_elites_scheduler_nightly.yaml` (`iterations: 650`, `steps: 200` in the nightly entrypoint, surrogate off, buffer `artifacts/surrogate/buffer_nightly.jsonl`) → (2) `scripts/train_surrogate.py` → `artifacts/surrogate/checkpoints/nightly.pkl` → (3) surrogate phase with `map_elites_scheduler_nightly_surrogate.yaml`, resuming baseline archive (~3h full pipeline on a ~140 ms/eval host). Outputs under `artifacts/map_elites_nightly/baseline/` and `.../surrogate/` plus `nightly_pipeline_summary.json`. `--single-run` runs one phase; production scheduler remains `map_elites_scheduler.yaml` (`iterations: 10000`).

File: `worldspace/illuminators/loop.py`

`run_iteration(config, archive, rng, counters, emitter, grid_size=..., steps=..., jsonl_path=None)` processes one batch: for `candidate_id` in `0 .. batch_size-1`, resolves the emitter, selects a target bin, calls `CandidateEmitter.emit`, runs `evaluate_candidate`, then `insert_evaluated` or `insert_and_persist`, and increments `RunCounters`. `run_scheduler` repeats for `config.iterations`. Batch tie-break: strict `>` insert plus fixed slot order means equal fitness in the same bin within one iteration keeps the first accepted elite.

Package `worldspace/illuminators/emitters/`: `CandidateEmitter` protocol; `RandomEmitter` (independent random worlds, §8.7); `GeneticEmitter` (uniform crossover + Gaussian mutation on 21-gene encoding, parent selection §8.8, `mutation_scale` from scheduler YAML); `LlmEmitter` (LLM JSON → `WorldSpec`, invalid → one `RandomWalkWorldGenerator` step from parent 1, `emitter_type` `llm` or `llm_fallback`); `MapElitesEmitter` dispatches by resolved slot kind (`resolve_emitter_for_slot` maps disabled LLM slots to `random` before `emit`); `StubCandidateEmitter` remains for simple tests. Emitters return specs with `seed=0` and canonical `cell_types`; `evaluate_candidate` assigns the canonical hash seed.

LLM prompts live under repository `prompts/` and load via `worldspace/prompt_files.read_prompt`. MAP-Elites: `map_elites_llm_emitter_system.txt`, `map_elites_llm_emitter_user.txt` (`render_system_prompt`, `build_user_prompt`; `system_prompt_version()` hashes the system file). Legacy generators: `llm_patch_system.txt`, `llm_patch_local_goal.txt`, `llm_patch_global_goal.txt`, `llm_patch_global_rules.txt`, `llm_patch_instruction.txt`, `llm_patch_output_format.json`, `llm_hybrid_local_user.txt`, `llm_vision_system.txt`, `llm_vision_user.txt`. Parsing: `worldspace/specs/world_spec_from_llm.py`. Shared field bounds: `worldspace/specs/world_spec_constraints.py`.

### Math helpers

File: `worldspace/math.py`

Shared numeric routines used by the simulator, metrics, and pipeline (neighbor counts, Lloyd k-means on memmap rows, entropy/oscillation/diversity helpers). Imported as `from . import math as ws_math` to avoid clashing with the standard library `math` module.

### Metrics (World Coordinates in Behavior Space)

File: `worldspace/metrics.py`

`WorldMetrics` holds the behavioral coordinates (filled by `run_world`). There are **`METRICS_VECTOR_DIM` (= 12)** entries in `as_vector()` / JSON metrics:

- `entropy`, `stability`, `average_lifespan`, `density_mean`, `oscillation_score`, `diversity`
- `mo_eoc_indicator` — **Multi-Objective + Edge-of-Chaos** scalar; see `multi_objective_edge_of_chaos_indicator` in `metrics.py` and `docs/WORLDSPACE.md` §5.1.
- `topology_interface_index`, `topology_window_heterogeneity` — fast morphological / mesoscale proxies (`worldspace/math.py`).
- `compressibility_score` — zlib length vs raw `life‖food` bytes (description-length proxy).
- `ecology_state_entropy_norm`, `ecology_resource_adjacency` — joint `(life, food)` diversity and food–live spatial coupling.

Constants and vector layout live in `metrics.py` (`METRICS_VECTOR_DIM`).

### Space Construction (streaming)

File: `worldspace/pipeline.py`

`stream_world_space_to_jsonl(generator, n_worlds, path, k_clusters, echo_stdout=..., metrics_trace_path=..., ca_step_trace_path=...)`:

1. **Pass 1:** for each `WorldSpec` from `generator.iter_worlds(n)`, run `run_world` (passing CA trace file/handle into `run_world` when `ca_step_trace_path` is set). Append each final metrics vector to a **temporary float32 memmap** (`n × METRICS_VECTOR_DIM`). **Materialize** the list of `WorldSpec` objects for pass 2 (small structs; avoids a second `iter_worlds` pass, which halves remote LLM calls for `LLMWorldGenerator`).
2. Fit **dominant-metric + orthogonal sklearn PCA** on the memmap matrix (`_fit_dominant_metric_orthogonal_pca`): x-axis = deviation of the batch-highest-variance metric from its batch mean; y-axis = first PC of the remaining `METRICS_VECTOR_DIM - 1` columns (`sklearn.decomposition.PCA(n_components=1)`).
3. Run Lloyd k-means **row-wise** on the memmap (centroids `k × METRICS_VECTOR_DIM`, labels on a separate memmap).
4. **Pass 2:** for each index `i`, use cached `worlds[i]` plus memmap row `i`, project to 2D, assign `cluster_id`, emit one JSON line to the main output path (if any) and optionally stdout. If `metrics_trace_path` is set, write the same space record plus `yield_index` per line to that file (suitable for `python -m worldspace.visualizer --metrics-jsonl ...` from CLI `--metrics-trace`).

Trace file handles are opened **inside** the same `try`/`finally` as the pipeline body so a failure opening the second trace file still closes the first.

RAM: memmap metrics stay **O(1)** vs batch size for the metric matrix; plus **O(n)** small `WorldSpec` objects for the cached list.

### Visualization (matplotlib + pandas for CA traces)

Package: `worldspace/visualizer/` — Matplotlib uses the **`Agg`** backend (file output only).

- **`plotting.py`**: `plot_dominant_metric_delta_scatter_from_jsonl`, `plot_world_metrics_pca_scatter_from_jsonl`, `plot_world_metrics_umap_scatter_from_jsonl`, `plot_simulation_final_grid`; pandas helpers `load_ca_step_trace_jsonl`, `summarize_ca_step_trace_by_world`; `plot_ca_step_metrics_timeseries`, `plot_ca_step_pca_trajectories`, `plot_ca_step_umap_trajectories` for `--ca-step-trace` JSONL.
- **`visualizer.py`** + **`__main__.py`**: run as `python -m worldspace.visualizer` with **`--output-dir`** (required) and optional **`--metrics-jsonl`** (writes `dominant_metric_delta.png`, `pca.png`, `umap.png`) and/or **`--ca-step-jsonl`** (writes `ca_step_timeseries.png`, `pca_trajectories.png`, `umap_trajectories.png`). Optional **`--ca-trace-worlds`**, **`--metrics`**, **`--summary`** for CA plots (see `docs/WORLDSPACE.md` §6.1, §8).

Public re-exports also live on `worldspace` (from `worldspace.visualizer`).

### Generator Configs (YAML)

Files:

- `worldspace/specs/genetic_world_generator.yaml`
- `worldspace/specs/llm_world_generator.yaml`
- `worldspace/specs/hybrid_world_generator.yaml`
- `worldspace/specs/neural_world_generator.yaml`
- `worldspace/specs/map_elites_scheduler.yaml` (MAP-Elites illuminator iteration mix and limits)
- `worldspace/specs/map_elites_scheduler_mini.yaml` (fast CI / reproducibility scheduler)

These files define default generator behavior and provider/model routing; the CLI flag `--generator-spec PATH` overrides the path when using `--generator genetic|llm|hybrid|neural` (the YAML shape is validated for that generator).

## CLI and Output Artifacts

Files: `worldspace/cli.py`, `worldspace/__main__.py`

Run:

`python -m worldspace --generator random --worlds 30 --steps 200 --grid 40`

Other generator modes:

- `--generator genetic --generator-spec ...`
- `--generator llm --generator-spec ...`
- `--generator hybrid --generator-spec ...`
- `--generator neural --generator-spec ...`

Optional persistence (JSONL only, one JSON object per line):

- `--metrics-trace PATH` — JSONL: one line per world after dominant-metric-delta layout + k-means (`yield_index`, `world`, `metrics`, `dominant_metric_delta_xy`, `dominant_metric_delta_axis_labels`, `cluster_id`). See `docs/WORLDSPACE.md` §6.1.

Optional copy of each full space record to stdout (no `yield_index` in those lines):

- `--echo-lines`

Optional sidecars (any `--generator`):

- `--ca-step-trace PATH` — JSONL: one line per CA timestep for each **pipeline** `run_world` (`yield_index`, `ca_step`, `metrics`). Does not trace extra `run_world` calls inside generators (e.g. LLM parent scoring).

Unified visualizer (fixed filenames under `--output-dir`):

`python -m worldspace.visualizer --output-dir results/plots --metrics-jsonl results/trace.jsonl --ca-step-jsonl results/ca_steps.jsonl --ca-trace-worlds 0,10,20 --summary`

If neither `--metrics-trace` nor `--echo-lines` is set, no per-world JSONL is written by the CLI (CA-step-only runs stay quiet on stdout).

## Separation from Legacy Stack

This architecture is intentionally independent of disabled integrations (`celery`, `redis`, `Plot`).  
It is designed as an offline research pipeline first, with optional future adapters for web/streaming.
