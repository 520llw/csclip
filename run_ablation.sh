#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/ablation_study.py > /tmp/ablation_output.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/ablation_output.txt
