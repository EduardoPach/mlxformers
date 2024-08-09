# Variables
PYTHON := $(shell which pdm > /dev/null 2>&1 && echo "pdm run" || echo "python -m")
SRC_DIR := src/mlxformers

install:
	if command -v pdm >/dev/null 2>&1; then \
		pdm install --dev; \
	else \
		pip install -e ".[dev]"; \
	fi

lint:
	$(PYTHON) ruff check $(SRC_DIR)

lint-fix:
	$(PYTHON) ruff check --fix $(SRC_DIR)

style:
	$(PYTHON) ruff format $(SRC_DIR)

.PHONY: install lint fix
