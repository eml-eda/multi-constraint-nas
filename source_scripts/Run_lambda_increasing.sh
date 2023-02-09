#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

_model=$1
_gumbel=$2
gpu=$3
_elements=$4
_loss=$5
ID=$6
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
_epochs=500
_l=$7
_size_target=( 2.0e+04 4.0e+04 6.0e+04 )
_cd_size=( 1.0e-02 1.0e-03 1.0e-04 1.0e-05 1.0e-06 1.0e-07 )

_latency_target_20=( 2.0e+06 3.0e+06 3.5e+06 4.0e+06 5.0e+06 )
_cd_ops=( 5.0e-4 1.0e-4 5.0e-5 1.0e-5 5.0e-6 1.0e-6 1.0e-7 )
for m in "${!_size_target[@]}"
do
    for n in "${!_cd_size[@]}"
    do
        python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size[n]} --cd-ops ${_cd_ops[0]} --size-target ${_size_target[m]} --latency-target ${_latency_target_20[0]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_last/srch_icl_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[n]}_${_latency_target_20[0]}_${_cd_ops[0]}_${_l}_${_elements}_${_loss}.log
    done
done