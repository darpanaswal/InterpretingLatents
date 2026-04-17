#!/bin/bash
# > runs/logit_lens.txt
source reason/bin/activate
# ProsQA (K=6 for all)
# python -m experiments.probe_thoughts.logit_lens --task prosqa --model base --k 6
# python -m experiments.probe_thoughts.logit_lens --task prosqa --model cot --k 6
# python -m experiments.probe_thoughts.logit_lens --task prosqa --model pause --k 6
# python -m experiments.probe_thoughts.logit_lens --task prosqa --model coconut --k 6
# python -m experiments.probe_thoughts.logit_lens --task prosqa --model coconut_u --k 6

# # GSM (K=3 for Coconut-family, K=6 for CODI)
# python -m experiments.probe_thoughts.logit_lens --task gsm --model base --k 6
# python -m experiments.probe_thoughts.logit_lens --task gsm --model cot --k 6
# python -m experiments.probe_thoughts.logit_lens --task gsm --model pause --k 6
# python -m experiments.probe_thoughts.logit_lens --task gsm --model coconut --k 6
# python -m experiments.probe_thoughts.logit_lens --task gsm --model coconut_u --k 6
python -m experiments.probe_thoughts.logit_lens --task gsm --model codi --k 6