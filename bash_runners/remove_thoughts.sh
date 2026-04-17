#!/bin/bash
> runs/remove_thoughts.txt
source reason/bin/activate
python -u -m experiments.remove_thoughts --task prosqa --models pause coconut coconut_u
python -u -m experiments.remove_thoughts --task gsm --models pause coconut coconut_u codi