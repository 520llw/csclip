#!/bin/bash
cd /home/xut/csclip
/data/software/mamba/envs/cel/bin/python experiments/ten_shot_classify.py > /tmp/10shot_output.txt 2>&1
echo "EXIT=$?" >> /tmp/10shot_output.txt
