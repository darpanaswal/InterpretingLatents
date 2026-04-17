#!/bin/bash
> runs/control.txt
source reason/bin/activate
python -u -m experiments.bfs_control.superposition_control --mode base
python -u -m experiments.bfs_control.superposition_control --mode cot
python -u -m experiments.bfs_control.superposition_control --mode pause
python -u -m experiments.bfs_control.superposition_control --mode coconut
python -u -m experiments.bfs_control.superposition_control --mode coconut_u