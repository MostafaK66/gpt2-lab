.PHONY: install test lint format typecheck build quality

install:
	python -m pip install -e ".[dev,plot]"

test:
	python -m pytest --cov=gpt2lab --cov-report=term-missing

lint:
	python -m ruff check src tests

format:
	python -m ruff format src tests

typecheck:
	python -m mypy

build:
	python -m build

quality: lint typecheck test build
