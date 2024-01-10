#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

_model="PIT"
_gumbel="False"
gpu="0"
_elements="mem"
_loss="max"
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
_epochs=500
_size_target=2.0e+04
_cd_size=1.0e-03
_latency_target=4.0e+06
_cd_ops=1.0e-4

python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size} --cd-ops ${_cd_ops} --size-target ${_size_target} --latency-target ${_latency_target} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} 
#python kws_training.py --epochs ${_epochs} --cd-size ${_cd_size} --cd-ops ${_cd_ops} --size-target ${_size_target} --latency-target ${_latency_target} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} 
#python vww_training.py --epochs ${_epochs} --cd-size ${_cd_size} --cd-ops ${_cd_ops} --size-target ${_size_target} --latency-target ${_latency_target} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} 
#python tiny_imagenet_training.py --epochs ${_epochs} --cd-size ${_cd_size} --cd-ops ${_cd_ops} --size-target ${_size_target} --latency-target ${_latency_target} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} 
