#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

model="PIT"
_benchmark="tiny"
gpu="0"
_gumbel="True"
_elements="mem_constraint+lat_constraint"
_loss="max"
ID="Results"
_l="increasing"
_hardware="None"

    _latency_target=( 2.5e+15 )
    _cd_ops=( 5.0e-8 )
    echo "tiny benchmark"
    _epochs=50
    m=0

    echo "Run on GPU 0"
    export CUDA_VISIBLE_DEVICES=0
    _size_target=( 8.46e+06 )
    _cd_size=( 1.2e-05 )
    python compute_energy_saving.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${model} 