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
_l="increasing"
_size_target=( 1.93e+04 3.87e+04 5.8e+04 )
_cd_size=( 2.4e-04 3.6e-04 7.2e-04 )

_latency_target=( 2.5e+07 2.5e+07 2.5e+07 )
_cd_ops=( 5.0e-8 5.0e-8 5.0e-8 )

for m in "${!_size_target[@]}"
do
    python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_lambda_increasing/srch_icl_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
done

_l="const"
_size_target=( 1.93e+04 3.87e+04 5.8e+04 )
_cd_size=( 1.0e-02 1.0e-03 1.0e-04 1.0e-05 1.0e-06 1.0e-07 )

for m in "${!_size_target[@]}"
do
    for n in "${!_cd_size[@]}"
    do
        python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size[n]} --cd-ops ${_cd_ops[0]} --size-target ${_size_target[m]} --latency-target ${_latency_target[0]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_lambda_increasing/srch_icl_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[n]}_${_latency_target[0]}_${_cd_ops[0]}_${_l}_${_elements}_${_loss}.log
    done
done

