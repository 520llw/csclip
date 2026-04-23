#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/cellpose_benchmark.py > /tmp/cellpose_bench.txt 2>&1
echo "EXIT=$?" >> /tmp/cellpose_bench.txt
