#!/bin/bash
export PATH=/research/huang/workspaces/hytopot/miniconda3/bin:$PATH
source activate program1
python3 run.py $@
