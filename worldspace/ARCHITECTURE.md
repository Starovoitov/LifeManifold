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

`run_world(world, *, ca_step_trace_file=None, ca_step_trace_yield_index=0) -> SimulationResult`

Optional keyword-only args (used by the pipeline when `--ca-step-trace` is set): after each CA timestep, append one JSON line (`yield_index`, `ca_step`, `metrics`) to `ca_step_trace_file`. Other callers omit them.

Simulation model:

- 2D grid
- Moore neighborhood
- Birth/survival update
- Noise flips
- Predation deaths
- Resource regeneration and feed bonus

Returns `SimulationResult` with **`metrics: WorldMetrics`**, optional **`final_life`** / **`final_food`** for plotting and diagnostics.  
Inside the loop there are **no growing Python lists** of per-step grids or full density series: online statistics, a bounded density window for oscillation, and one final grid snapshot.

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
