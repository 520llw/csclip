#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/iterative_optimize.py > /tmp/optimize_output.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/optimize_output.txt
