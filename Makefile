UV ?= uv

.PHONY: install install-dashboard activate pylint fix smoke-map-elites nightly-map-elites

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

pylint:
	$(UV) run ruff check .
	$(UV) run pyright .
	$(UV) run black --check .

fix:
	$(UV) run ruff check . --fix
	$(UV) run black .

smoke-map-elites:
	$(UV) run python -m unittest tests.test_map_elites_smoke -v

nightly-map-elites:
	$(UV) run python -m worldspace.scripts.run_map_elites_nightly
