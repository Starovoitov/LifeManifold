UV ?= uv

.PHONY: install activate pylint fix

install:
	$(UV) venv
	$(UV) sync --all-groups
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
