PYTHON ?= python3

.PHONY: verify lint test install

install:
	$(PYTHON) -m pip install -e '.[upscale,dev]'

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest

verify: lint test
