# LifeManifold Streamlit Dashboard

Interactive research UI for MAP-Elites archives, surrogate predictions, and metrics.

Design spec: `artifacts/STREAMLIT_DASHBOARD_TZ_v1.0.md`  
Task breakdown: `artifacts/STREAMLIT_DASHBOARD_EPICS_AND_TASKS_v1.0.md`

## Data sources

The dashboard reads **JSONL** archives and buffers (plus `.pkl` checkpoints and `.json` run summaries). It does not use Parquet as a project format.

Default paths are in `config/config.yaml` (relative to the repository root).

## Setup

From the repository root, with the project venv:

```bash
make install          # core + dev
uv sync --group dashboard
```

Or only dashboard extras: `uv sync --group dashboard` (requires `pandas`, `pyyaml`, `lightgbm` from the default sync).

Alternative: `pip install -r dashboard/requirements.txt`

## Run

From the `dashboard/` directory (recommended):

```bash
cd dashboard
streamlit run Home.py
```

Entry scripts call `path_setup.install_paths()` before `import dashboard` so the repo root is on `PYTHONPATH`. If you see `ModuleNotFoundError: No module named 'dashboard'`, run from repo root instead:

```bash
cd /path/to/LifeManifold
PYTHONPATH=. streamlit run dashboard/Home.py
```

Smoke data for local development: `artifacts/map_elites_smoke/map_elites_archive.jsonl`.

## Layout

| Path | Role |
|------|------|
| `Home.py` | Overview and navigation |
| `pages/` | Multi-page app (Archive, Surrogate, Metrics, LLM, Buffer) |
| `components/` | Loaders, charts, filters (E1+) |
| `utils/config.py` | YAML config and repo path resolver |
| `utils/plotting.py` | Shared Plotly dark theme |
