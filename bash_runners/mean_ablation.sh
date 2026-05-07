#!/bin/bash
> runs/mean_ablation.txt
# source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task prosqa --model pause
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task prosqa --model coconut 
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task prosqa --model coconut_u
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task prosqa --model codi
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task gsm    --model pause
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task gsm    --model coconut
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task gsm    --model coconut_u
python -u -m experiments.geometry.mean_ablation --n_gpus 4 --n_thoughts 6 --intervention core --task gsm    --model codi