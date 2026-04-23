#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/ten_shot_v2.py > /tmp/10shot_v2_output.txt 2>&1
echo "EXIT=$?" >> /tmp/10shot_v2_output.txt
