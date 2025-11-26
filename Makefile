# Makefile at project root
.PHONY: test bench_gnn bench_scheduler all

test:
	pytest -q

bench_gnn:
	python -m scripts.eval_gnn_vs_baseline --num-seeds 5 --num-packets 5000

bench_scheduler:
	python -m scripts.eval_scheduler_scaling --num-requests 50000

all: test bench_gnn bench_scheduler
