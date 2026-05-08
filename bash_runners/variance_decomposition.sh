#!/bin/bash
> runs/variance_decomposition.txt
# source reason/bin/activate
source primitive/bin/activate

python -u -m experiments.geometry.variance_decomposition --all