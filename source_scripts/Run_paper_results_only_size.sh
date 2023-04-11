#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ..

_model=$1
_benchmark=$2
gpu=$3
_gumbel="True"
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

if [[ $_benchmark == "tiny" ]]; then

    echo "tiny benchmark"
    _epochs=50
    if [[ $_model == "PIT" ]]; then
        _size_target=( 8.46e+06 5.64e+06 2.82e+06 )
        _cd_size=( 1.2e-06 6.6e-07 5.4e-07 )
    fi

    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 2.5e+15 2.5e+15 2.5e+15 )
    _cd_ops=( 5.0e-8 5.0e-8 5.0e-8 )

    for m in "${!_size_target[@]}"
    do
        python tiny_imagenet_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_tiny_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done

fi

if [[ $_benchmark == "icl" ]]; then

    echo "icl benchmark"
    _epochs=500
    if [[ $_model == "PIT" ]]; then
        _size_target=( 1.93e+04 3.87e+04 5.8e+04 )
        _cd_size=( 2.4e-04 3.6e-04 7.2e-04 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 1.61e+04 3.55e+04 5.48e+04 )
        _cd_size=( 2.29e-03 3.34e-03 6.21e-03 )
    fi

    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 2.5e+07 2.5e+07 2.5e+07 )
    _cd_ops=( 5.0e-8 5.0e-8 5.0e-8 )

    for m in "${!_size_target[@]}"
    do
        python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_supernet_val/srch_icl_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done

fi

if [[ $_benchmark == "kws" ]]; then

    echo "kws benchmark"
    _epochs=200
    if [[ $_model == "PIT" ]]; then
        _size_target=( 5.50e+03 1.1e+04 1.65e+04 )
        _cd_size=( 9.0e-05 1.3e-04 2.6e-04 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 2.18e+03 7.68e+03 1.32e+04 )
        _cd_size=( 5.04e-02 6.98e-02 1.13e-01 )
    fi
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 1.0e+07 1.0e+07 1.0e+07 )
    _cd_ops=( 5.0e-8 5.0e-8 5.0e-8 )
    for m in "${!_size_target[@]}"
    do
        python kws_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_supernet_val/srch_kws_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done
fi

if [[ $_benchmark == "vww" ]]; then

    echo "vww benchmark"
    _epochs=70
    if [[ $_model == "PIT" ]]; then
        _size_target=( 1.30e+04 2.7e+04 5.3e+04 1.07e+05 )
        _cd_size=( 2.3e-05 2.5e-05 2.95e-05 4.4e-05 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 5.27e+04 1.06e+05 )
        _cd_size=( 6.21e-04 9.30e-04 )
    fi
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 1.5e+07 1.5e+07 1.5e+07 1.5e+07 )
    _cd_ops=( 1.5e-8 1.5e-8 1.5e-8 1.5e-8 )

    for m in "${!_size_target[@]}"
    do
        python vww_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_supernet_val/srch_vww_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done

fi

if [[ $_benchmark == "amd" ]]; then

    echo "amd benchmark"
    _epochs=100
    _size_target=( 0.33e+05 0.67e+05 1.34e+05 2.01e+05 )
    _cd_size=( 9.4e-02 2.46e-03 9.12e-03 1.12e-02 )
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 5.4e+05 5.4e+05 5.4e+05 5.4e+05 )
    _cd_ops=( 0.5e-07 0.5e-07 0.5e-07 0.5e-07 )

    for m in "${!_size_target[@]}"
    do
        python amd_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_supernet_val/srch_amd_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log

    done

fi