#!/bin/bash
# > runs/extract_thoughts.txt
source reason/bin/activate
python -u -m experiments.probe_thoughts.extract_thoughts --task prosqa --model pause     --n_thoughts 6
python -u -m experiments.probe_thoughts.extract_thoughts --task prosqa --model coconut   --n_thoughts 6
python -u -m experiments.probe_thoughts.extract_thoughts --task prosqa --model coconut_u --n_thoughts 6
python -u -m experiments.probe_thoughts.extract_thoughts --task gsm --model pause        --n_thoughts 6
python -u -m experiments.probe_thoughts.extract_thoughts --task gsm --model coconut      --n_thoughts 6
python -u -m experiments.probe_thoughts.extract_thoughts --task gsm --model coconut_u    --n_thoughts 6
python -u -m experiments.probe_thoughts.extract_thoughts --task gsm --model codi         --n_thoughts 6