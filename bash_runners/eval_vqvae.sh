#!/bin/bash
> runs/eval_vqvae.txt
source reason/bin/activate
python -u -m experiments.vqvae.eval_vqvae \
    --codebook_paths \
        outputs/vqvae/coconut/codebook_K1.pt \
        outputs/vqvae/coconut/codebook_K2.pt \
        outputs/vqvae/coconut/codebook_K3.pt \
        outputs/vqvae/coconut/codebook_K4.pt \
        outputs/vqvae/coconut/codebook_K8.pt \
        outputs/vqvae/coconut/codebook_K16.pt \
        outputs/vqvae/coconut/codebook_K32.pt \
        outputs/vqvae/coconut/codebook_K64.pt \
        outputs/vqvae/coconut/codebook_K128.pt \
        outputs/vqvae/coconut/codebook_K256.pt \
    --mode both \
    --bfs_categories_path outputs/recursionControl/bfs_categories_coconut.json
python -u -m experiments.vqvae.eval_vqvae \
    --codebook_paths \
        outputs/vqvae/coconut_u/codebook_K1.pt \
        outputs/vqvae/coconut_u/codebook_K2.pt \
        outputs/vqvae/coconut_u/codebook_K3.pt \
        outputs/vqvae/coconut_u/codebook_K4.pt \
        outputs/vqvae/coconut_u/codebook_K8.pt \
        outputs/vqvae/coconut_u/codebook_K16.pt \
        outputs/vqvae/coconut_u/codebook_K32.pt \
        outputs/vqvae/coconut_u/codebook_K64.pt \
        outputs/vqvae/coconut_u/codebook_K128.pt \
        outputs/vqvae/coconut_u/codebook_K256.pt \
    --mode both \
    --bfs_categories_path outputs/recursionControl/bfs_categories_coconut_u.json \
    --model coconut_u
python -u -m experiments.vqvae.eval_vqvae \
    --codebook_paths \
        outputs/vqvae/pause/codebook_K1.pt \
        outputs/vqvae/pause/codebook_K2.pt \
        outputs/vqvae/pause/codebook_K3.pt \
        outputs/vqvae/pause/codebook_K4.pt \
        outputs/vqvae/pause/codebook_K8.pt \
        outputs/vqvae/pause/codebook_K16.pt \
        outputs/vqvae/pause/codebook_K32.pt \
        outputs/vqvae/pause/codebook_K64.pt \
        outputs/vqvae/pause/codebook_K128.pt \
        outputs/vqvae/pause/codebook_K256.pt \
    --mode both \
    --bfs_categories_path outputs/recursionControl/bfs_categories_pause.json \
    --model pause