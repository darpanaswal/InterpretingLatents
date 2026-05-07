EXP_NAME="gradient_subspace_geometry"

oarsub -p "network_address='lig-gpu5.imag.fr'" \
       -l /host=1/gpu=1,walltime=12:00:00 \
       -O "runs/${EXP_NAME}.txt" \
       -E "runs/${EXP_NAME}.txt" \
       "./bash_runners/${EXP_NAME}.sh"