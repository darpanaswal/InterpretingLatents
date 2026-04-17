#!/bin/bash
> runs/INLP.txt
source reason_pyt/bin/activate

# Stage 2: Ablation (Loads precomputed INLP if available, otherwise computes and saves)
# python -u -m experiments.amnesic_probing.inlp_classification --model pause
# python -u -m experiments.amnesic_probing.inlp_classification --model coconut
# python -u -m experiments.amnesic_probing.inlp_classification --model coconut_u

# python -u -m experiments.amnesic_probing.inlp_regression --ridge_alpha 100 --model pause
# python -u -m experiments.amnesic_probing.inlp_regression --ridge_alpha 100 --model coconut
# python -u -m experiments.amnesic_probing.inlp_regression --ridge_alpha 100 --model coconut_u
# python -u -m experiments.amnesic_probing.inlp_regression --ridge_alpha 100 --model codi