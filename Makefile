.PHONY: test lint format migrate serve

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff format .
	python -m ruff check --fix .

migrate:
	python -m alembic upgrade head

serve:
	lpg serve
