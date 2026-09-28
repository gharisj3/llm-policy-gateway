.PHONY: test lint format

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff format .
	python -m ruff check --fix .
