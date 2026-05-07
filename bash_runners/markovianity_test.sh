#!/bin/bash
> runs/markovianity_test.txt
# source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.geometry.markovianity_test --task prosqa --model all --num_gpus 4 --project_to_subspace both
python -u -m experiments.geometry.markovianity_test --task gsm    --model all --num_gpus 4 --project_to_subspace both