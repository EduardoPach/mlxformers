# Variables
PYTHON := $(shell which pdm > /dev/null 2>&1 && echo "pdm run" || echo "python -m")
SRC_DIR := src/mlxformers
TEST_DIR := tests

install:
	if command -v pdm >/dev/null 2>&1; then \
		pdm install --dev; \
	else \
		pip install -e ".[dev]"; \
	fi

lint:
	$(PYTHON) ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix:
	$(PYTHON) ruff check --fix $(SRC_DIR) $(TEST_DIR)

style:
	$(PYTHON) ruff format $(SRC_DIR) $(TEST_DIR)

publish:
	@pdm publish -u $(PYPI_USERNAME) -P $(PYPI_PASSWORD)

.PHONY: install lint fix publish
