.PHONY: install test lint type-check format clean run

install:
	pip install -e ".[dev]"

test:
	pytest --cov=src --cov-report=term-missing -v

lint:
	ruff check src/ tests/

type-check:
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ .coverage htmlcov

run:
	uvicorn src.server:app --reload --port 8005
