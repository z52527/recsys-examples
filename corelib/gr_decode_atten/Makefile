SHELL := /bin/bash
PYTHONPATH := PYTHONPATH=.

.PHONY: setup tt vt bb profile clean

# Install all dependencies
setup:
	pip install -r requirements.txt

# Quick unit tests
tt:
	$(PYTHONPATH) python tests/test_context.py
	$(PYTHONPATH) python tests/test_beam.py
	$(PYTHONPATH) python tests/test_fwd.py

# Full parametrized test suite (1200 cases)
vt:
	$(PYTHONPATH) pytest tests/test_context.py tests/test_beam.py tests/test_fwd.py -v

# Benchmark: 3-kernel vs fused, dn=1..16
bb:
	$(PYTHONPATH) python tests/benchmark.py --mode benchmark

# Profile: single run for ncu/nsys capture
profile:
	$(PYTHONPATH) python tests/benchmark.py --mode profile --decode_nums 3

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true
	rm -rf .pytest_cache .hypothesis
