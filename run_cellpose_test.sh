#!/bin/bash
cd /home/xut/csclip
export PYTHONUNBUFFERED=1
/data/software/mamba/envs/cel/bin/python -u experiments/cellpose_benchmark.py > /tmp/cp_bench_result.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/cp_bench_result.txt
