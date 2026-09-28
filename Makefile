.PHONY: test lint

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

