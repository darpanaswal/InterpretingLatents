#!/bin/bash
# > runs/random_corruption.txt
source reason/bin/activate

# Stage 1: Random Corruption
# python -u -m experiments.amnesic_probing.random_corruption --task prosqa --model pause
# python -u -m experiments.amnesic_probing.random_corruption --task prosqa --model coconut
# python -u -m experiments.amnesic_probing.random_corruption --task prosqa --model coconut_u

# python -u -m experiments.amnesic_probing.random_corruption --task gsm --model pause
# python -u -m experiments.amnesic_probing.random_corruption --task gsm --model coconut
python -u -m experiments.amnesic_probing.random_corruption --task gsm --model coconut_u
python -u -m experiments.amnesic_probing.random_corruption --task gsm --model codi