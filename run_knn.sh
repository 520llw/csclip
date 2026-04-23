#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/knn_refine.py > /tmp/knn_output.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/knn_output.txt
