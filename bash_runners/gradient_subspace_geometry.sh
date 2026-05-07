#!/bin/bash
> runs/gradient_subspace_geometry.txt
source primitive/bin/activate

python -u -m experiments.geometry.gradient_subspace_geometry --task prosqa --model pause
python -u -m experiments.geometry.gradient_subspace_geometry --task prosqa --model coconut
python -u -m experiments.geometry.gradient_subspace_geometry --task prosqa --model coconut_u
python -u -m experiments.geometry.gradient_subspace_geometry --task prosqa --model codi
python -u -m experiments.geometry.gradient_subspace_geometry --task gsm    --model pause
python -u -m experiments.geometry.gradient_subspace_geometry --task gsm    --model coconut
python -u -m experiments.geometry.gradient_subspace_geometry --task gsm    --model coconut_u
python -u -m experiments.geometry.gradient_subspace_geometry --task gsm    --model codi