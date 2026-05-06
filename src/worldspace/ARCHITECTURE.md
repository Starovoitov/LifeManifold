# WorldSpace MVP Architecture

This document describes the new "world space" architecture for the project.

## Goal

Represent each cellular automata world as a point in parameter space, simulate behavior, extract behavior metrics, and map worlds into a searchable/comparable embedding.

Pipeline:

`Generator -> WorldSpec JSON -> Simulator -> Metrics -> Embedding/Clustering -> World Space`

## Core Concepts

### WorldSpec

File: `src/worldspace/spec.py`

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

File: `src/worldspace/generators.py`

Implemented generator levels:

1. `random_walk(...)` utility (small local parameter changes)
2. `RandomWorldGenerator` (independent random worlds)
3. `RandomWalkWorldGenerator` (trajectory in world space)
4. Markov hierarchy:
   - `MarkovWorldGenerator` (base class)
   - `TwoStateNoiseMarkovGenerator` (calm/chaos switching)
   - `RuleBiasMarkovGenerator` (survival/reproduction bias)
5. `GeneticWorldGenerator` (selection + mutation on `interestingness` fitness)
6. Future stubs:
   - `NeuralWorldGenerator` (placeholder)
   - `LLMWorldGenerator` (placeholder)

### Simulator

File: `src/worldspace/simulator.py`

`run_world(world: WorldSpec) -> SimulationResult`

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

File: `src/worldspace/math.py`

Shared numeric routines used by the simulator, metrics, and pipeline (neighbor counts, PCA projection, k-means, entropy/oscillation/diversity helpers). Imported inside the package as `from . import math as ws_math` to avoid clashing with the Python standard library `math` module.

### Metrics (World Coordinates in Behavior Space)

File: `src/worldspace/metrics.py`

`WorldMetrics` holds the behavioral coordinates (filled by `run_world`). There are **`METRICS_VECTOR_DIM` (= 7)** entries in `as_vector()` / JSON metrics:

- `entropy`, `stability`, `average_lifespan`, `density_mean`, `oscillation_score`, `diversity`
- `interestingness` — favors high entropy, stability, and diversity; subtracts **extinction penalty** `clamp(1 - final_density, 0, 1)` (empty final grid ⇒ penalty 1).

Constants and vector layout live in `metrics.py` (`METRICS_VECTOR_DIM`).

### Space Construction (streaming)

File: `src/worldspace/pipeline.py`

`stream_world_space_to_jsonl(generator, n_worlds, path, k_clusters, echo_stdout=...)`:

1. **Pass 1 — streaming over worlds:** for each `WorldSpec` from `generator.iter_worlds(n)`, run `run_world`, write the metrics vector (`METRICS_VECTOR_DIM` floats) to a **temporary float32 memmap** row, and update PCA sufficient statistics (`sum_x`, `sum_xx`) in **O(1)** extra RAM.
2. Fit PCA mean + 2D basis from sufficient statistics (`math.pca_mean_and_basis_2d`).
3. Run Lloyd k-means **row-wise** on the memmap (centroids are `k × METRICS_VECTOR_DIM`, labels on a separate memmap).
4. **Pass 2 — streaming again:** same `iter_worlds` order, read each metrics row from disk, project to 2D, assign `cluster_id`, **append one JSON line** to the output file (and optionally print it).

No Python list of all worlds or all `SpacePoint` objects is kept. RAM vs. batch size is **O(1)** (plus fixed small buffers and k-means state); disk holds the temporary `(n × METRICS_VECTOR_DIM)` metrics file only until the function returns.

### Visualization (matplotlib)

File: `src/worldspace/viz.py`

Matplotlib is confined to this submodule and uses the **`Agg`** backend (file output only).

- `plot_world_embedding(points, path, ...)` — scatter from in-memory point-like objects (`embedding_2d`, `cluster_id`).
- `plot_world_embedding_from_jsonl(jsonl_path, path, ...)` — scatter from a saved JSONL run (used by the CLI).
- `plot_simulation_final_grid(result, path, ...)` — heatmap of `result.final_life`.

CLI: `--plot` reads JSONL from `--output`, or from a temporary JSONL if `--output` is omitted (file removed after plotting).

## CLI and Output Artifacts

Files: `src/worldspace/cli.py`, `src/worldspace/__main__.py`

Run:

`python -m src.worldspace --generator random --worlds 30 --steps 200 --grid 40`

Optional persistence (JSONL only, one JSON object per line):

- `--output results/world_space.jsonl`

Optional copy of each line to stdout while writing a file:

- `--echo-lines`

Optional figure (requires `--output` first):

- `--plot results/world_space_map.png`

If `--output` is omitted, each JSON record is printed as one line to stdout (JSONL on stdout) with the same streaming memory profile.

## Separation from Legacy Stack

This architecture is intentionally independent of disabled integrations (`celery`, `redis`, `Plot`).  
It is designed as an offline research pipeline first, with optional future adapters for web/streaming.
