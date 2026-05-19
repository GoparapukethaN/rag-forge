.PHONY: verify test lint benchmark-sample

PYTHON ?= python

verify: test lint

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check rag_forge tests

benchmark-sample:
	./scripts/run-sample-benchmark.sh
