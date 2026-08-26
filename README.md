# LifeManifold

**Explore cellular-automata “worlds” as points in rule space** — simulate them, search diverse high-quality niches with MAP-Elites, optionally guide generation with an LLM + surrogate, and inspect archives in a Streamlit UI.

Core loop: `WorldSpec` (rules + parameters) → `run_world` → behavioral metrics → archive or metrics JSONL.

Requires **Python ≥ 3.11** and [`uv`](https://docs.astral.sh/uv/).

## Install

```bash
make install                 # creates .venv and syncs dependencies
source .venv/bin/activate    # optional; or prefix commands with uv run
```

Dashboard extras are included in `make install`. To sync only the dashboard group later: `make install-dashboard`.

## Quickstart (MAP-Elites)

Primary path — illuminate a behavior archive from a scheduler YAML:

```bash
uv run python -m worldspace --illuminator mapelites \
  --scheduler worldspace/specs/map_elites_scheduler_mini.yaml \
  --output-dir artifacts/map_elites_smoke \
  --seed 42 --steps 200 --grid 8
```

Writes `artifacts/map_elites_smoke/map_elites_archive.jsonl`. Same smoke as CI:

```bash
make smoke-map-elites
```

## Dashboard

```bash
streamlit run dashboard/Home.py
```

Default scan picks up the smoke archive above. Paths and discovery: `dashboard/config/config.yaml` — see [docs/DASHBOARD.md](docs/DASHBOARD.md).

## Batch exploration (legacy)

Generate and simulate a fixed set of worlds (PCA / k-means layout, not the MAP-Elites archive):

```bash
uv run python -m worldspace \
  --generator random \
  --worlds 30 \
  --steps 200 \
  --grid 40 \
  --metrics-trace results/metrics.jsonl
```

## Documentation

| Start here | |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Package map, two CLI paths |
| [MAP-Elites](docs/MAPELITES.md) | Illuminator, archives, schedulers |
| [WorldSpace](docs/WORLDSPACE.md) | Simulator, `WorldSpec`, generators |
| [Surrogate](docs/SURROGATE_MODEL.md) | Hints, buffer, acquisition (`shadow` / `filter`) |
| [Dashboard](docs/DASHBOARD.md) | Streamlit setup and pages |

| Also | |
| --- | --- |
| [Domains](docs/DOMAINS.md) | Maze / dungeon / sphere runners (not main CLI) |
| [Formulas](docs/FORMULAS.md) | Metrics and fitness definitions |
| [Protocol freeze timeline](artifacts/PROTOCOL_FREEZE_TIMELINE.md) | Lock / extension / reporting ledger (not in the journal PDF) |

## License

[MIT](LICENSE) — © 2026 Artem Starovoitov
