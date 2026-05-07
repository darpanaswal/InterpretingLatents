#!/bin/bash
> runs/remove_thoughts.txt
# source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.thought_causality.remove_thoughts --task prosqa 
python -u -m experiments.thought_causality.remove_thoughts --task gsm