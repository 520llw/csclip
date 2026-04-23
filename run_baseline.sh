#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/baseline_diagnosis.py > /tmp/baseline_output.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/baseline_output.txt
