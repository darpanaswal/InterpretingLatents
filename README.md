# Observable Patterns Are Not Explanations

🎉 **Accepted at EMNLP 2026 (Main Conference)!**

Code to reproduce the experiments in *Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models*. [Paper (arXiv)](https://arxiv.org/pdf/2606.12689)

## Setup

```bash
pip install -r requirements.txt
```

Set `HUGGINGFACE_API_KEY` in a `.env` file at the repo root if you need to pull models/datasets from the Hub (only `helpers/hf_transfer.py` requires it).

Before running any experiment, train or download the target model and its controls. All trained checkpoints (GPT-2 and Llama-3.2-1B-Instruct, all tasks and controls) are available at the [Hugging Face collection](https://hf.co/collections/darpanaswal/latent-reasoning-models). See also [Appendix D: Models, Controls, and Training](#appendix-d-models-controls-and-training) below.

Some experiments (§6 Markovianity, and several extended analyses) require pre-extracted latent thoughts:

```bash
python -u -B -m experiments.extract_thoughts \
    --task prosqa \
    --model_family gpt2 \
    --model pause \
    --split both \
    --n_thoughts 6 \
    --n_gpus 1
```

---

## § 4 — Observable Structures Are Insufficient for Mechanistic Attribution

### Case Study 1: Superposition and BFS-like search (Graph-Hopping)

```bash
python -u -m experiments.dead_salmon.superposition \
    --model_family gpt2 \
    --mode base        # base | cot | pause | coconut | coconut_u | codi
```

Run once per `--mode` (all six controls/models) for each `--model_family`.

### Case Study 2: Scratchpad Thinking via Logit-Lens (Arithmetic-Reasoning)

```bash
python -u -m experiments.dead_salmon.logit_lens \
    --model_family gpt2 \
    --task gsm \
    --model pause \
    --k 6
```

Run once per `--model` for each `--model_family`.

### Plot Figure 1 (both case studies)

```bash
python helpers/plot_epiphenomena.py \
    --out_dir Plots/epiphenomena \
    --tables_dir Tables/statistical \
    --extended_tables_dir Tables/extended
```

---

## § 5 — When and How do LRMs use Latent Thoughts?

### Latent Thought Ablation at Test-Time (Table 1)

```bash
torchrun --nproc_per_node=1 -m experiments.ablation.remove_thoughts \
    --model_family gpt2 \
    --task prosqa \
    --models pause      # pause | coconut | coconut_u | codi
```

### Per-Position Causal Tracing (Figure 2)

```bash
python -u -m experiments.causal_tracing.causal_trace \
    --task prosqa \
    --model_family gpt2 \
    --model pause \
    --n_gpus 1 \
    --granularity single \
    --batch_size 64 \
    --max_instances 2000
```

### Plot Figure 2 / causal-tracing heatmaps

```bash
python helpers/plot_causal_trace.py \
    --mode restorative \
    --granularity single \
    --out_fig_dir Plots/causal_trace \
    --out_tex_dir Tables/statistical
```

### Gradient-Subspace Interventions (Figure 3)

First fit the gradient subspace, then run the intervention sweep:

```bash
python -u -m experiments.ablation.gradient_subspace \
    --task prosqa \
    --model_family gpt2 \
    --model pause

python -u -m experiments.ablation.gradient_subspace_interventions \
    --task prosqa \
    --model_family gpt2 \
    --model pause \
    --n_gpus 1
```

### Plot Figure 3

```bash
python helpers/plot_gradient_subspace_interventions.py \
    --out_tables_dir Tables/statistical
```

---

## § 6 — The Dynamics and Geometry of Latent Thoughts

### Markovianity of Thought Trajectories (Figure 4)

```bash
python -u -B -m experiments.geometry.markovianity_test \
    --task prosqa \
    --model pause \
    --model_family gpt2 \
    --project_to_subspace gold \
    --n_gpus 1
```

### Plot Figure 4

```bash
python helpers/plot_markov.py --tables_dir Tables/statistical
```

### Geometric Stability of Gradient-Subspaces (Figure 5)

```bash
python -u -m experiments.geometry.gradient_subspace_geometry \
    --task prosqa \
    --model pause \
    --model_family gpt2 \
    --subspace_source gold
```

### Plot Figure 5

```bash
python helpers/plot_gradient_geometry.py
```

---

## Extended Analyses

The following reproduce Appendix A (Llama-3.2-1B-Instruct extension of §4–§6, same commands as above with `--model_family llama`) and Appendix B (additional analyses). Run at your discretion — these receive secondary emphasis in the paper.

### Predicted-token gradient subspace

```bash
python -u -m experiments.ablation.gradient_subspace_predtoken \
    --task prosqa \
    --model_family gpt2 \
    --model pause \
    --max_new 8

python -u -m experiments.ablation.gradient_subspace_interventions \
    --task prosqa \
    --model_family gpt2 \
    --model pause \
    --n_gpus 1 \
    --bases_path outputs/gradient_geometry_predtoken/gpt2/prosqa/pause/bases.npz \
    --output_dir outputs/gradient_subspace_interventions_predtoken
```

### Angle between gold and predicted-token subspaces

```bash
python -u -B -m experiments.geometry.gold_predtoken_angles \
    --task prosqa \
    --model_family gpt2 \
    --model pause
```

Plot: `python helpers/plot_gold_pred_angle.py` (writes `Tables/extended/gold_pred_angle_<family>.tex`, no CLI flags).

### Gradient-subspace dimensionality tables

```bash
python helpers/plot_subspace_ranks.py
```

### Mean-ablations and isolations (variance-component interventions, Figure 19)

```bash
python -u -m experiments.geometry.mean_ablation \
    --task prosqa \
    --model coconut_u \
    --model_family gpt2 \
    --n_thoughts 6 \
    --n_gpus 1
```

Plot:

```bash
python helpers/plot_mean_ablation.py \
    --out_fig_dir Plots/mean_ablation \
    --out_main_dir Tables/main \
    --out_stats_dir Tables/extended
```

### Variance decomposition (§ Appendix B, static organization of latent thoughts)

```bash
python -u -m experiments.geometry.variance_decomposition \
    --task prosqa \
    --model pause \
    --model_family gpt2
```

Or, for every (task, model) pair in one run:

```bash
python -u -m experiments.geometry.variance_decomposition --all --model_family gpt2
```

### Random-corruption robustness check

```bash
python -u -m experiments.random_corruption \
    --task prosqa \
    --model_family gpt2 \
    --model pause \
    --n_gpus 1 \
    --codi_batch_size 8 \
    --n_seeds 3
```

Plot: `python helpers/plot_remove_thoughts.py --out_main_dir Tables/main --out_stats_dir Tables/statistical` (also covers Table 1/Table 2 test-time ablation).

---

## Statistical Tables (Appendix C)

Full bootstrap CIs and McNemar significance tables are generated as a byproduct of the plotting scripts above (written to `Tables/statistical/` and `Tables/extended/`). No separate reproduction steps are provided here.

---

## Appendix D: Models, Controls, and Training

Models are trained separately per `(dataset, model_family, model)` combination. Hyperparameters are listed in the paper (Tables 29–30, Appendix D.3); training scripts are not included in this cleanup. Download checkpoints from the [Hugging Face collection](https://hf.co/collections/darpanaswal/latent-reasoning-models) or train your own, and place them under `model/<task>/<model_family>/<model>/` (see `src/config.py` for exact paths) before running any experiment above.

---

## Citation

```bibtex
@article{aswal2026observable,
  title={Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models},
  author={Aswal, Darpan and Ferraz, Thomas Palmeira and Zhou, Yongxin and Peyrard, Maxime},
  journal={arXiv preprint arXiv:2606.12689},
  year={2026}
}
```
