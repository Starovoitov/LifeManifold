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
5. Future stubs:
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

Outputs time-series and history needed for analysis:

- density per step
- alive cells per step
- death ages
- frame history

### Math helpers

File: `src/worldspace/math.py`

Shared numeric routines used by the simulator, metrics, and pipeline (neighbor counts, PCA projection, k-means, entropy/oscillation/diversity helpers). Imported inside the package as `from . import math as ws_math` to avoid clashing with the Python standard library `math` module.

### Metrics (World Coordinates in Behavior Space)

File: `src/worldspace/metrics.py`

`compute_metrics(result) -> WorldMetrics` returns:

- `entropy`
- `stability`
- `average_lifespan`
- `density_mean`
- `oscillation_score`
- `diversity`

These 6 values are the behavioral coordinates of a world.

### Space Construction

File: `src/worldspace/pipeline.py`

`explore_world_space(generator, n_worlds, k_clusters)`:

1. Generate world specs.
2. Simulate each world.
3. Compute metric vectors.
4. Reduce to 2D via PCA (`math.pca_2d` in `src/worldspace/math.py`).
5. Group similar worlds with simple k-means (`math.kmeans`).

Output: `list[SpacePoint]` where each point contains:

- original `world`
- computed `metrics`
- `embedding_2d`
- `cluster_id`

## CLI and Output Artifacts

Files: `src/worldspace/cli.py`, `src/worldspace/__main__.py`

Run:

`python -m src.worldspace --generator random --worlds 30 --steps 200 --grid 40`

Optional persistence (JSONL only, one JSON object per line):

- `--output results/world_space.jsonl`

The CLI always prints a JSON array to stdout and can additionally write the same records as JSONL to a file.

## Separation from Legacy Stack

This architecture is intentionally independent of disabled integrations (`celery`, `redis`, `Plot`).  
It is designed as an offline research pipeline first, with optional future adapters for web/streaming.
