#!/bin/bash
> runs/gradient_subspace.txt
# source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.thought_causality.gradient_subspace  --task prosqa --model pause
python -u -m experiments.thought_causality.gradient_subspace  --task prosqa --model coconut
python -u -m experiments.thought_causality.gradient_subspace  --task prosqa --model coconut_u
python -u -m experiments.thought_causality.gradient_subspace  --task prosqa --model codi
python -u -m experiments.thought_causality.gradient_subspace  --task gsm    --model pause     
python -u -m experiments.thought_causality.gradient_subspace  --task gsm    --model coconut    
python -u -m experiments.thought_causality.gradient_subspace  --task gsm    --model coconut_u     
python -u -m experiments.thought_causality.gradient_subspace  --task gsm    --model codi   