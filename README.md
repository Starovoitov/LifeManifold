# LifeManifold

**Explore cellular-automata “worlds” as points in rule space** — simulate them, search diverse high-quality niches with MAP-Elites, optionally guide generation with an LLM + surrogate, and inspect archives in a Streamlit UI.

Core loop: `WorldSpec` (rules + parameters) → `run_world` → behavioral metrics → archive or metrics JSONL.

Requires **Python ≥ 3.11** and [`uv`](https://docs.astral.sh/uv/), or Docker (the path below). The shell launchers in `scripts/` are bash, and `make install` builds a Linux virtualenv.

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

## Run experiments in Docker

Use this on Windows, or anywhere the bash launchers should run in Linux. Docker Desktop has to be running. The first run builds `lifemanifold-experiment` from `uv.lock`, including the CUDA PyTorch wheel, so the image is several GB. That environment is copied once into a Docker volume mounted at `/workspace/.venv`. The repo itself is bind-mounted at `/workspace`, so archives and logs stay in this tree, and the `.venv` on the host is left alone. A `.env` in the repo root is loaded when the file exists.

From PowerShell:

```powershell
.\scripts\docker-experiment.ps1 scripts/run_cvt_baseline.sh
.\scripts\docker-experiment.ps1 uv run python -m worldspace --illuminator mapelites --scheduler worldspace/specs/map_elites_scheduler_mini.yaml --output-dir artifacts/map_elites_smoke --seed 42 --steps 200 --grid 8
.\scripts\docker-experiment.ps1 -Gpu uv run python scripts/train_surrogate.py
.\scripts\docker-experiment.ps1
```

A path that ends in `.sh` is passed to `bash`. No arguments opens a shell in the container. `-Gpu` adds [`docker-compose.gpu.yml`](docker-compose.gpu.yml) and needs an NVIDIA GPU visible to Docker Desktop. Every other run stays on CPU.

From a Unix shell the same service is:

```bash
docker compose run --rm experiment bash scripts/run_cvt_baseline.sh
docker compose run --rm -it experiment bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm experiment uv run python scripts/train_surrogate.py
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
