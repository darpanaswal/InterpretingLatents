#!/bin/bash
> runs/interventions.txt
# source reason/bin/activate
source primitive/bin/activate

# Stage 3: Steering (Uses precomputed concept_to_idx for verification to skip INLP)
python -u -m experiments.amnesic_probing.inlp_interventions --task prosqa --model pause --n_gpus 4
python -u -m experiments.amnesic_probing.inlp_interventions --task prosqa --model coconut --n_gpus 4
python -u -m experiments.amnesic_probing.inlp_interventions --task prosqa --model coconut_u --n_gpus 4
python -u -m experiments.amnesic_probing.inlp_interventions --task prosqa --model codi --n_gpus 4

python -u -m experiments.amnesic_probing.inlp_interventions --task gsm --model pause --n_gpus 4
python -u -m experiments.amnesic_probing.inlp_interventions --task gsm --model coconut --n_gpus 4
python -u -m experiments.amnesic_probing.inlp_interventions --task gsm --model coconut_u --n_gpus 4
python -u -m experiments.amnesic_probing.inlp_interventions --task gsm --model codi --n_gpus 4