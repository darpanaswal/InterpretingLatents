#!/bin/bash
> runs/superposition.txt
# source reason/bin/activate
source primitive/bin/activate

python -m experiments.dead_salmon.superposition --mode base
python -m experiments.dead_salmon.superposition --mode cot
python -m experiments.dead_salmon.superposition --mode pause
python -m experiments.dead_salmon.superposition --mode coconut
python -m experiments.dead_salmon.superposition --mode coconut_u
python -m experiments.dead_salmon.superposition --mode codi