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
5. `GeneticWorldGenerator` (PyGAD-backed evolution on `interestingness`)
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

Returns `SimulationResult` with **`metrics: WorldMetrics`** and **`final_life`** for optional plotting.  
Inside the loop there are **no growing Python lists** of per-step grids or full density series: online statistics, a bounded density window for oscillation, and one final grid snapshot.

### Math helpers

File: `worldspace/math.py`

Shared numeric routines used by the simulator, metrics, and pipeline (neighbor counts, Lloyd k-means on memmap rows, entropy/oscillation/diversity helpers). Imported as `from . import math as ws_math` to avoid clashing with the standard library `math` module.

### Metrics (World Coordinates in Behavior Space)

File: `worldspace/metrics.py`

`WorldMetrics` holds the behavioral coordinates (filled by `run_world`). There are **`METRICS_VECTOR_DIM` (= 7)** entries in `as_vector()` / JSON metrics:

- `entropy`, `stability`, `average_lifespan`, `density_mean`, `oscillation_score`, `diversity`
- `interestingness` — favors high entropy, stability, and diversity; subtracts **extinction penalty** `clamp(1 - final_density, 0, 1)` (empty final grid ⇒ penalty 1).

Constants and vector layout live in `metrics.py` (`METRICS_VECTOR_DIM`).

### Space Construction (streaming)

File: `worldspace/pipeline.py`

`stream_world_space_to_jsonl(generator, n_worlds, path, k_clusters, echo_stdout=..., metrics_trace_path=..., ca_step_trace_path=...)`:

1. **Pass 1:** for each `WorldSpec` from `generator.iter_worlds(n)`, run `run_world` (passing CA trace file/handle into `run_world` when `ca_step_trace_path` is set). Append each final metrics vector to a **temporary float32 memmap** (`n × METRICS_VECTOR_DIM`). **Materialize** the list of `WorldSpec` objects for pass 2 (small structs; avoids a second `iter_worlds` pass, which halves remote LLM calls for `LLMWorldGenerator`). Optionally append one JSON line per world to `metrics_trace_path` (`yield_index`, `world`, `metrics`).
2. Fit **dominant-metric + orthogonal sklearn PCA** on the memmap matrix (`_fit_dominant_metric_orthogonal_pca`): x-axis = deviation of the batch-highest-variance metric from its batch mean; y-axis = first PC of the remaining six columns (`sklearn.decomposition.PCA(n_components=1)`).
3. Run Lloyd k-means **row-wise** on the memmap (centroids `k × METRICS_VECTOR_DIM`, labels on a separate memmap).
4. **Pass 2:** for each index `i`, use cached `worlds[i]` plus memmap row `i`, project to 2D, assign `cluster_id`, emit one JSON line to the main output (and optionally stdout).

Trace file handles are opened **inside** the same `try`/`finally` as the pipeline body so a failure opening the second trace file still closes the first.

RAM: memmap metrics stay **O(1)** vs batch size for the metric matrix; plus **O(n)** small `WorldSpec` objects for the cached list.

### Visualization (matplotlib + pandas for CA traces)

Package: `worldspace/visualizer/` — Matplotlib uses the **`Agg`** backend (file output only).

- **`plotting.py`**: `plot_world_embedding`, `plot_world_embedding_from_jsonl`, `plot_simulation_final_grid`; pandas helpers `load_ca_step_trace_jsonl`, `summarize_ca_step_trace_by_world`; `plot_ca_step_metrics_timeseries`, `plot_ca_step_pca_trajectories` for `--ca-step-trace` JSONL.
- **`visualizer.py`** + **`__main__.py`**: run as `python -m worldspace.visualizer` with subcommands:
  - **`embedding`** — scatter from main pipeline JSONL (`embedding ... <jsonl> --plot <png>`).
  - **`ca-trace`** — time-series + PCA figures from `--ca-step-trace` output (see `docs/WORLDSPACE.md` §8).

Public re-exports also live on `worldspace` (from `worldspace.visualizer`).

### Generator Configs (YAML)

Files:

- `worldspace/specs/genetic_world_generator.yaml`
- `worldspace/specs/llm_world_generator.yaml`
- `worldspace/specs/hybrid_world_generator.yaml`
- `worldspace/specs/neural_world_generator.yaml`

These files define default generator behavior and provider/model routing; CLI flags can override paths to these configs.

## CLI and Output Artifacts

Files: `worldspace/cli.py`, `worldspace/__main__.py`

Run:

`python -m worldspace --generator random --worlds 30 --steps 200 --grid 40`

Other generator modes:

- `--generator genetic --genetic-spec ...`
- `--generator llm --llm-spec ...`
- `--generator hybrid --hybrid-spec ...`
- `--generator neural --neural-spec ...`

Optional persistence (JSONL only, one JSON object per line):

- `--output results/world_space.jsonl`

Optional copy of each line to stdout while writing a file:

- `--echo-lines`

Optional sidecars (any `--generator`):

- `--metrics-trace PATH` — JSONL: one line per simulated world from pass 1 (`yield_index`, `world`, `metrics`; no embedding/cluster).
- `--ca-step-trace PATH` — JSONL: one line per CA timestep for each **pipeline** `run_world` (`yield_index`, `ca_step`, `metrics`). Does not trace extra `run_world` calls inside generators (e.g. LLM parent scoring).

Embedding scatter (reads JSONL written with `--output`):

`python -m worldspace.visualizer embedding results/world_space.jsonl --plot results/world_space_map.png`

CA trace plots (pandas + matplotlib):

`python -m worldspace.visualizer ca-trace results/ca_steps.jsonl --output-dir results/ca_plots --worlds 0,10,20 --summary`

If `--output` is omitted, main JSONL lines are **not** printed unless `--echo-lines` is set (trace-only runs stay quiet).

## Separation from Legacy Stack

This architecture is intentionally independent of disabled integrations (`celery`, `redis`, `Plot`).  
It is designed as an offline research pipeline first, with optional future adapters for web/streaming.
