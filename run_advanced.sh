#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/advanced_optimize.py > /tmp/advanced_output.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/advanced_output.txt
