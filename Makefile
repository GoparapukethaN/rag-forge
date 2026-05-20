.PHONY: verify test lint benchmark-sample sample-check

PYTHON ?= python

verify: test lint

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check rag_forge tests

benchmark-sample:
	./scripts/run-sample-benchmark.sh

sample-check:
	./scripts/verify-sample.sh
