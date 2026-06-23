UV ?= uv

.PHONY: install install-dashboard activate lint fix smoke-map-elites smoke-map-elites-cvt nightly-map-elites github-llm-map-elites github-llm-map-elites-full surrogate-artifacts surrogate-artifacts-quick surrogate-acquisition-baseline migrate-surrogate-buffer calibrate-surrogate

install:
	$(UV) venv
	$(UV) sync --all-groups

install-dashboard:
	$(UV) sync --group dashboard
	@echo ""
	@echo "Environment ready."
	@echo "Activate it with: source .venv/bin/activate"

activate:
	@echo "Run: source .venv/bin/activate"

lint:
	$(UV) run ruff check .
	$(UV) run pyright .
	$(UV) run black --check .

fix:
	$(UV) run ruff check . --fix
	$(UV) run black .

smoke-map-elites:
	$(UV) run python -m unittest tests.test_map_elites_smoke -v

smoke-map-elites-cvt:
	$(UV) run python -m unittest tests.test_map_elites_smoke.TestMapElitesSmoke.test_mini_cvt_scheduler_smoke_leaves_artifacts -v

nightly-map-elites:
	$(UV) run python -m worldspace.scripts.run_map_elites_nightly

# Qwen LLM + nightly_v2.pkl; default 120 iter (CI-safe). Fresh archive for more LLM lines.
github-llm-map-elites:
	$(UV) run python scripts/run_github_llm_map_elites.py --no-resume-nightly --train-surrogate-if-missing

# Full nightly-length LLM run (650 iter); too slow for default GHA 6h limit
github-llm-map-elites-full:
	$(UV) run python scripts/run_github_llm_map_elites.py --iterations 650 --no-resume-nightly --train-surrogate-if-missing

# Local surrogate buffer + checkpoints (synthetic data; artifacts/surrogate/ is gitignored)
surrogate-artifacts:
	$(UV) run python scripts/bootstrap_surrogate_artifacts.py

surrogate-artifacts-quick:
	$(UV) run python scripts/bootstrap_surrogate_artifacts.py --quick

# Backfill buffer from artifacts/map_elites_nightly, train latest.pkl, baseline manifest
surrogate-acquisition-baseline:
	$(UV) run python scripts/record_surrogate_acquisition_baseline.py --all-archive-lines --allow-quality-fail

migrate-surrogate-buffer:
	@echo "Example: uv run python scripts/migrate_surrogate_buffer.py --archive PATH --output artifacts/surrogate/buffer.jsonl --overwrite"

calibrate-surrogate:
	$(UV) run python scripts/calibrate_surrogate_uncertainty.py --allow-high-ece
