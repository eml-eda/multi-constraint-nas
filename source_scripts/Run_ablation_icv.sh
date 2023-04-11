#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

_model=$1
_benchmark=$2
gpu=$3
_gumbel="False"
_elements="mem_constraint+lat_constraint"
_loss="max"
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
if [[ $_benchmark == "icl" ]]; then

    echo "icl benchmark"
    _epochs=500
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 1.61e+04 1.61e+04 1.61e+04 1.61e+04 3.55e+04 3.55e+04 3.55e+04 3.55e+04 5.48e+04 5.48e+04 5.48e+04 5.48e+04 )
        _cd_size=( 2.29e-03 2.29e-03 2.29e-03 2.29e-03 3.34e-03 3.34e-03 3.34e-03 3.34e-03 6.21e-03 6.21e-03 6.21e-03 6.21e-03 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _latency_target=( 1.07e+06 2.14e+06 3.21e+6 2.5e+07 2.73e+06 5.46e+06 8.19e+06 2.5e+07 3.34e+06 6.68e+06 1.00e+07 2.5e+07 )
        _cd_ops=( 1.25e-05 1.41e-05 1.63e-05 5.0e-8 1.34e-05 1.65e-05 2.17e-05 5.0e-8 1.37e-05 1.77e-05 2.49e-05 5.0e-8 )
    fi

    for m in "${!_size_target[@]}"
    do
        python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_icv/srch_icl_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done

fi
