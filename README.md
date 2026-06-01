# LifeManifold

Cellular-automata world-space research: simulate worlds, run MAP-Elites quality–diversity search, and explore results in a Streamlit dashboard.

## Setup

```bash
make install
# optional: dashboard extras only
make install-dashboard
```

## Run a simulation

**Batch exploration** (generate worlds, simulate, write metrics JSONL):

```bash
uv run python -m worldspace \
  --generator random \
  --worlds 30 \
  --steps 200 \
  --grid 40 \
  --metrics-trace results/metrics.jsonl
```

**MAP-Elites** (scheduler YAML, archive JSONL under `output-dir`):

```bash
uv run python -m worldspace --illuminator mapelites \
  --scheduler worldspace/specs/map_elites_scheduler_mini.yaml \
  --output-dir artifacts/map_elites_smoke \
  --seed 42 --steps 200 --grid 8
```

Fast CI smoke (same as `make smoke-map-elites`):

```bash
make smoke-map-elites
```

## Visualization

**Streamlit dashboard** (archives, surrogate, metrics, acquisition log):

```bash
uv sync --group dashboard
streamlit run dashboard/Home.py
```

Point `dashboard/config/config.yaml` at your run output (default smoke archive: `artifacts/map_elites_smoke/map_elites_archive.jsonl`).

## More

| Topic | Doc |
|-------|-----|
| Simulator, generators, legacy pipeline | [docs/WORLDSPACE.md](docs/WORLDSPACE.md) |
| MAP-Elites | [docs/MAPELITES.md](docs/MAPELITES.md) |
| Dashboard | [docs/DASHBOARD.md](docs/DASHBOARD.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
