#!/usr/bin/env bash

size_target=$3
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd_size=$2
cd_ops=$7
latency_target=$4
l=$8
loss=$6
elements=$5
benchmark=$1
hardware=${10}
model=$9
if [[ "$1" == "icl" ]]; then
epochs=1
python icl_training.py --cd-size ${cd_size} --size-target ${size_target} --latency-target ${latency_target} --loss_elements ${elements} --loss_type ${loss} --cd-ops ${cd_ops} --epochs ${epochs} --l ${l} --model ${model} --hardware ${hardware} | tee log/srch_icl_${hardware}_${size_target}_${cd_ops}_${latency_target}_${cd_ops}_${l}_${elements}_${loss}.log
fi
if [[ "$1" == "vww" ]]; then
epochs=50
python vww_training.py --cd-size ${cd_size} --size-target ${size_target} --latency-target ${latency_target} --loss_elements ${elements} --loss_type ${loss} --cd-ops ${cd_ops} --epochs ${epochs} --l ${l} --model ${model} --hardware ${hardware} | tee log/srch_vww_${hardware}_${size_target}_${cd_ops}_${latency_target}_${cd_ops}_${l}_${elements}_${loss}.log
fi
if [[ "$1" == "kws" ]]; thenù
epochs=36
python kws_training.py --cd-size ${cd_size} --size-target ${size_target} --latency-target ${latency_target} --loss_elements ${elements} --loss_type ${loss} --cd-ops ${cd_ops} --epochs ${epochs} --l ${l} --model ${model} --hardware ${hardware} | tee log/srch_kws_${hardware}_${size_target}_${cd_ops}_${latency_target}_${cd_ops}_${l}_${elements}_${loss}.log
fi