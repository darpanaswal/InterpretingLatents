#!/bin/bash
> runs/scaffold_tests.txt
source reason/bin/activate
# source primitive/bin/activate


# ── PE ablation: all models × both tasks × both modes ──
for TASK in prosqa gsm; do
    for MODEL in base cot codi; do
    # for MODEL in base cot pause coconut coconut_u codi; do
        for MODE in zero constant random_gaussian random_shuffle; do
            python -u -m experiments.probe_thoughts.pe_ablation \
                --task $TASK --model $MODEL --mode $MODE
        done
    done
done

# ── Noise-input test: all models × both tasks ──
for TASK in prosqa gsm; do
    for MODEL in base cot codi; do
    # for MODEL in base cot pause coconut coconut_u codi; do
        python -u -m experiments.probe_thoughts.noise_input_test \
            --task $TASK --model $MODEL
    done
done