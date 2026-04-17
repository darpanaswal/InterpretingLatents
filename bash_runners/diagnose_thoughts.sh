#!/bin/bash
> runs/diagnose_thoughts.txt
source reason/bin/activate

python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task prosqa --model pause
python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task prosqa --model coconut
python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task prosqa --model coconut_u
python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model pause
python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model coconut
python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model coconut_u
python -m experiments.probe_thoughts.diagnose_thoughts --reverse_inlp_iter 3 --task gsm    --model codi