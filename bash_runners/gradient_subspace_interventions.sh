#!/bin/bash
> runs/gradient_subspace_interventions.txt
# source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.thought_causality.gradient_subspace_interventions --task prosqa --n_gpus 4 --model pause
python -u -m experiments.thought_causality.gradient_subspace_interventions --task prosqa --n_gpus 4 --model coconut
python -u -m experiments.thought_causality.gradient_subspace_interventions --task prosqa --n_gpus 4 --model coconut_u
python -u -m experiments.thought_causality.gradient_subspace_interventions --task prosqa --n_gpus 4 --model codi
python -u -m experiments.thought_causality.gradient_subspace_interventions --task gsm    --n_gpus 4 --model pause
python -u -m experiments.thought_causality.gradient_subspace_interventions --task gsm    --n_gpus 4 --model coconut
python -u -m experiments.thought_causality.gradient_subspace_interventions --task gsm    --n_gpus 4 --model coconut_u
python -u -m experiments.thought_causality.gradient_subspace_interventions --task gsm    --n_gpus 4 --model codi