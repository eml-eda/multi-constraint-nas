#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

_model=$1
_benchmark=$2
gpu=$3
_gumbel="True"
_elements="mem_constraint+lat_constraint"
_loss="abs"
ID="Results"
_l="increasing"
_hardware="None"
if [[ $gpu == "0" ]]; then
    echo "Run on GPU 0"
    export CUDA_VISIBLE_DEVICES=0
fi
if [[ $gpu == "1" ]]; then
    echo "Run on GPU 1"
    export CUDA_VISIBLE_DEVICES=1
fi
if [[ $gpu == "2" ]]; then
    echo "Run on GPU 2"
    export CUDA_VISIBLE_DEVICES=2
fi
if [[ $gpu == "3" ]]; then
    echo "Run on GPU 3"
    export CUDA_VISIBLE_DEVICES=3
fi

if [[ $_benchmark == "kws" ]]; then

    echo "kws benchmark"
    _epochs=200
    if [[ $_model == "PIT" ]]; then
        _size_target=( 5.50e+03 5.50e+03 5.50e+03 5.50e+03 1.1e+04 1.1e+04 1.1e+04 1.1e+04 1.65e+04 1.65e+04 1.65e+04 1.65e+04 )
        _cd_size=( 9.0e-05 9.0e-05 9.0e-05 9.0e-05 1.3e-04 1.3e-04 1.3e-04 1.3e-04 2.6e-04 2.6e-04 2.6e-04 2.6e-04 )
    fi
    
    if [[ $_model == "PIT" ]]; then
        _latency_target=( 1.65e+05 3.29e+05 4.94e+05 1.0e+07 2.40e+05 4.80e+05 7.20e+05 1.0e+07 2.57e+05 5.13e+05 7.70e+05 1.0e+07 )
        _cd_ops=( 6.02e-06 6.44e-06 6.4e-06 5.0e-8 6.21e-06 6.89e-06 7.75e-06 5.0e-8 6.25e-06 7.00e-06 7.95e-06 5.0e-8 )
    fi

    for m in "${!_size_target[@]}"
    do
        python kws_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_ablation_abs/srch_kws_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done
fi