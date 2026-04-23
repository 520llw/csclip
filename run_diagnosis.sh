#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/deep_diagnosis.py > /tmp/diagnosis_output.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/diagnosis_output.txt
