#!/bin/bash
> runs/INLP.txt
# source reason/bin/activate
source primitive/bin/activate

# Stage 2: Ablation (Loads precomputed INLP if available, otherwise computes and saves)
python -u -m experiments.amnesic_probing.inlp --model pause
python -u -m experiments.amnesic_probing.inlp --model coconut
python -u -m experiments.amnesic_probing.inlp --model coconut_u
python -u -m experiments.amnesic_probing.inlp --model codi

python -u -m experiments.amnesic_probing.inlp --ridge_alpha 100 --model pause
python -u -m experiments.amnesic_probing.inlp --ridge_alpha 100 --model coconut
python -u -m experiments.amnesic_probing.inlp --ridge_alpha 100 --model coconut_u
python -u -m experiments.amnesic_probing.inlp --ridge_alpha 100 --model codi