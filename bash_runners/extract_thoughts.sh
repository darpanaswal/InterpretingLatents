#!/bin/bash
> runs/extract_thoughts.txt
# source reason/bin/activate
source primitive/bin/activate

python -m experiments.extract_thoughts --task prosqa --model base      --n_thoughts 6
python -m experiments.extract_thoughts --task prosqa --model cot       --n_thoughts 6
python -m experiments.extract_thoughts --task prosqa --model pause     --n_thoughts 6
python -m experiments.extract_thoughts --task prosqa --model coconut   --n_thoughts 6
python -m experiments.extract_thoughts --task prosqa --model coconut_u --n_thoughts 6
python -m experiments.extract_thoughts --task prosqa --model codi      --n_thoughts 6

python -m experiments.extract_thoughts --task gsm --model base         --n_thoughts 6
python -m experiments.extract_thoughts --task gsm --model cot          --n_thoughts 6
python -m experiments.extract_thoughts --task gsm --model pause        --n_thoughts 6
python -m experiments.extract_thoughts --task gsm --model coconut      --n_thoughts 6
python -m experiments.extract_thoughts --task gsm --model coconut_u    --n_thoughts 6
python -m experiments.extract_thoughts --task gsm --model codi         --n_thoughts 6