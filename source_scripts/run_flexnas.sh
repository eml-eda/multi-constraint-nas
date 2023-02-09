#!/usr/bin/env bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8

_benchmark=$1
_cd_size=$2
_size_target=$3
_latency_target=$4
_elements=$5
_loss=$6
_cd_ops=$7
_l=$8
_model=$9
_hardware=${10}
if [[ "$1" == "icl" ]]; then
_epochs=500
python icl_training.py --cd-size ${_cd_size} --size-target ${_size_target} --latency-target ${_latency_target} --loss_elements ${_elements} --loss_type ${_loss} --cd-ops ${_cd_ops} --epochs ${_epochs} --l ${_l} --model ${_model} --hardware ${_hardware} | tee log/srch_icl_tmp_const_${_model}_${_hardware}_${_size_target}_${_cd_size}_${_latency_target}_${_cd_ops}_${_l}_${_elements}_${_loss}.log
fi
if [[ "$1" == "vww" ]]; then
_epochs=50
python vww_training.py --cd-size ${cd_size} --size-target ${size_target} --latency-target ${latency_target} --loss_elements ${elements} --loss_type ${loss} --cd-ops ${cd_ops} --epochs ${epochs} --l ${l} --model ${model} --hardware ${hardware} | tee log/srch_vww_${model}_${hardware}_${size_target}_${cd_ops}_${latency_target}_${cd_ops}_${l}_${elements}_${loss}.log
fi
if [[ "$1" == "kws" ]]; then
_epochs=36
python kws_training.py --cd-size ${cd_size} --size-target ${size_target} --latency-target ${latency_target} --loss_elements ${elements} --loss_type ${loss} --cd-ops ${cd_ops} --epochs ${epochs} --l ${l} --model ${model} --hardware ${hardware} | tee log/srch_kws_${model}_${hardware}_${size_target}_${cd_ops}_${latency_target}_${cd_ops}_${l}_${elements}_${loss}.log
fi