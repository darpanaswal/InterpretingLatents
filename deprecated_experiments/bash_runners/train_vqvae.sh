#!/bin/bash
> runs/train_vqvae.txt
source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.vqvae.train_vqvae \
        --model coconut \
        --thoughts_path outputs/thoughts/thoughts_coconut.pt
python -u -m experiments.vqvae.train_vqvae \
        --model coconut_u \
        --thoughts_path outputs/thoughts/thoughts_coconut_u.pt
python -u -m experiments.vqvae.train_vqvae \
        --model pause \
        --thoughts_path outputs/thoughts/thoughts_pause.pt