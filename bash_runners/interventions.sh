#!/bin/bash
> runs/interventions.txt
source reason_pyt/bin/activate

# Stage 3: Steering (Uses precomputed concept_to_idx for verification to skip INLP)
# python -u -m experiments.amnesic_probing.interventions_fast --task prosqa --model pause --n_gpus 4
# python -u -m experiments.amnesic_probing.interventions_fast --task prosqa --model coconut --n_gpus 4
# python -u -m experiments.amnesic_probing.interventions_fast --task prosqa --model coconut_u --n_gpus 4

# python -u -m experiments.amnesic_probing.interventions_fast --task gsm --model pause --n_gpus 4
python -u -m experiments.amnesic_probing.interventions_fast --task gsm --model coconut --max_instances 3
# python -u -m experiments.amnesic_probing.interventions_fast --task gsm --model coconut_u --n_gpus 4
# python -u -m experiments.amnesic_probing.interventions_fast --task gsm --model codi --n_gpus 4