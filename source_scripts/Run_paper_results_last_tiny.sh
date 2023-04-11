#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

_model="PIT"
_benchmark="tiny"
_gumbel="True"
_elements="mem_constraint+lat_constraint"
_loss="max"
ID="Results"
_l="increasing"
_hardware="None"
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 2.5e+15 )
    _cd_ops=( 5.0e-8 )
    echo "tiny benchmark"
    _epochs=50
    m=0
    echo "Run on GPU 3"
    export CUDA_VISIBLE_DEVICES=3
    _size_target=( 8.46e+06 )
    _cd_size=( 1.2e-05 )
python tiny_imagenet_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_tiny_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log &
    

#     echo "Run on GPU 3"
#     export CUDA_VISIBLE_DEVICES=3
#     _size_target=( 10.9e+06 )
#     _cd_size=( 1.2e-05 )
# python tiny_imagenet_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_tiny_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log &
    
    echo "Run on GPU 1"
    export CUDA_VISIBLE_DEVICES=1
    _size_target=( 5.64e+06 )
    _cd_size=( 6.6e-06 )
python tiny_imagenet_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_tiny_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log &
    
    echo "Run on GPU 2"
    export CUDA_VISIBLE_DEVICES=2
    _size_target=( 2.82e+06 )
    _cd_size=( 5.4e-06 )
python tiny_imagenet_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_tiny_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log &
    

