#!/bin/bash
> runs/projected_thoughts.txt
source reason_pyt/bin/activate
# python -u -m experiments.amnesic_probing.projected_thoughts --task prosqa --model pause    
# python -u -m experiments.amnesic_probing.projected_thoughts --task prosqa --model coconut  
# python -u -m experiments.amnesic_probing.projected_thoughts --task prosqa --model coconut_u
python -u -m experiments.amnesic_probing.projected_thoughts --task gsm --model pause       
python -u -m experiments.amnesic_probing.projected_thoughts --task gsm --model coconut     
python -u -m experiments.amnesic_probing.projected_thoughts --task gsm --model coconut_u   
python -u -m experiments.amnesic_probing.projected_thoughts --task gsm --model codi        