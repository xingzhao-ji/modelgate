.PHONY: setup test demo

setup:
	python3.12 -m venv .venv
	.venv/bin/python -m pip install -r requirements.lock
	.venv/bin/python -m pip install --no-deps .

test:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	PYTHONPATH=src .venv/bin/pytest -q -W error

demo:
	.venv/bin/modelgate download
	.venv/bin/modelgate train --artifact artifacts/demo --report runs/demo/training.json
	.venv/bin/modelgate evaluate --artifact artifacts/demo --output runs/demo/evaluation.json
	.venv/bin/modelgate experiment --artifact artifacts/demo --output runs/demo/experiment.json
