#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

model="PIT"
_model="PIT"
_benchmark="tiny"
gpu="0"
_gumbel="True"
_elements="mem_constraint+lat_constraint"
_loss="max"
ID="Results"
_l="increasing"
_hardware="None"
percentage=$1
# if [[ $gpu == "0" ]]; then
#     echo "Run on GPU 0"
#     export CUDA_VISIBLE_DEVICES=0
# fi
# if [[ $gpu == "1" ]]; then
#     echo "Run on GPU 1"
#     export CUDA_VISIBLE_DEVICES=1
# fi
# if [[ $gpu == "2" ]]; then
#     echo "Run on GPU 2"
#     export CUDA_VISIBLE_DEVICES=2
# fi
# if [[ $gpu == "3" ]]; then
#     echo "Run on GPU 3"
#     export CUDA_VISIBLE_DEVICES=3
# fi
if [[ $_benchmark == "icl" ]]; then

    echo "icl benchmark"
    _epochs=500
    if [[ $_model == "PIT" ]]; then
        _size_target=( 1.93e+04 1.93e+04 1.93e+04 3.87e+04 3.87e+04 3.87e+04 5.8e+04 5.8e+04 5.8e+04 )
        _cd_size=( 2.4e-04 2.4e-04 2.4e-04 3.6e-04 3.6e-04 3.6e-04 7.2e-04 7.2e-04 7.2e-04 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 1.61e+04 1.61e+04 1.61e+04 3.55e+04 3.55e+04 3.55e+04 5.48e+04 5.48e+04 5.48e+04 )
        _cd_size=( 2.29e-03 2.29e-03 2.29e-03 3.34e-03 3.34e-03 3.34e-03 6.21e-03 6.21e-03 6.21e-03 )
    fi
    if [[ $_model == "PIT" ]]; then
        _latency_target=( 1.30e+06 2.61e+06 3.91e+06 2.02e+06 4.03e+06 6.05e+06 2.29e+06 4.59e+06 6.88e+06 )
        _cd_ops=( 1.25e-05 1.41e-05 1.63e-05 1.34e-05 1.65e-05 2.17e-05 1.37e-05 1.77e-05 2.49e-05 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _latency_target=( 1.07e+06 2.14e+06 3.21e+6 2.73e+06 5.46e+06 8.19e+06 3.34e+06 6.68e+06 1.00e+07 )
        _cd_ops=( 1.25e-05 1.41e-05 1.63e-05 1.34e-05 1.65e-05 2.17e-05 1.37e-05 1.77e-05 2.49e-05 )
    fi

    for m in "${!_size_target[@]}"
    do
        python icl_finetuning_GAP8.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${_model} | tee log/finetune_icl_${model}_${ID}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}.log
    done

fi

if [[ $_benchmark == "tiny" ]]; then

    # if [[ $_model == "PIT" ]]; then
    #     _size_target=( 8.46e+06 5.64e+06 2.82e+06 )
    #     _cd_size=( 1.2e-05 6.6e-06 5.4e-06 )
    # fi
    # if [[ $_model == "PIT" ]]; then
    # _latency_target=( 2.5e+15 2.5e+15 2.5e+15 )
    # _cd_ops=( 5.0e-8 5.0e-8 5.0e-8 )
    # fi

    # for m in "${!_size_target[@]}"
    # do
    #     python tiny_imagenet_finetuning_GAP8.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${_model} | tee log/finetune_tiny_${model}_${ID}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}.log
    # done

    _latency_target=( 2.5e+15 )
    _cd_ops=( 5.0e-8 )
    echo "tiny benchmark"
    _epochs=50
    m=0

    echo "Run on GPU 0"
    export CUDA_VISIBLE_DEVICES=0
    _size_target=( 8.46e+06 )
    _cd_size=( 1.2e-05 )
    python tiny_imagenet_finetuning_GAP8.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${_model} --percentage ${percentage} | tee log/finetune_tiny_${percentage}_${model}_${ID}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}.log &
    # python tiny_imagenet_finetuning_GAP8.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${_model} | tee log/finetune_tiny_${model}_${ID}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}.log

    echo "Run on GPU 1"
    export CUDA_VISIBLE_DEVICES=1
    _size_target=( 5.64e+06 )
    _cd_size=( 6.6e-06 )
    python tiny_imagenet_finetuning_GAP8.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${_model} --percentage ${percentage} | tee log/finetune_tiny_${percentage}_${model}_${ID}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}.log &

    echo "Run on GPU 2"
    export CUDA_VISIBLE_DEVICES=2
    _size_target=( 2.82e+06 )
    _cd_size=( 5.4e-06 )
    python tiny_imagenet_finetuning_GAP8.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --model ${_model} --percentage ${percentage} | tee log/finetune_tiny_${percentage}_${model}_${ID}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}.log &

fi
