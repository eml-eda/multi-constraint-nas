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

if [[ $_benchmark == "kws" ]]; then

    echo "kws benchmark"
    _epochs=200
    if [[ $_model == "PIT" ]]; then
        _size_target=( 5.50e+03 5.50e+03 5.50e+03 1.1e+04 1.1e+04 1.1e+04 1.65e+04 1.65e+04 1.65e+04 )
        _cd_size=( 9.0e-05 9.0e-05 9.0e-05 1.3e-04 1.3e-04 1.3e-04 2.6e-04 2.6e-04 2.6e-04 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 2.18e+03 7.68e+03 1.32e+04 )
        _cd_size=( 5.04e-02 6.98e-02 1.13e-01 )
    fi
    
    if [[ $_model == "PIT" ]]; then
        _latency_target=( 1.65e+05 3.29e+05 4.94e+05 2.40e+05 4.80e+05 7.20e+05 2.57e+05 5.13e+05 7.70e+05 )
        _cd_ops=( 6.02e-07 6.44e-07 6.4e-07 6.21e-07 6.89e-07 7.75e-07 6.25e-07 7.00e-07 7.95e-07 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _latency_target=( 1.0e+07 1.0e+07 1.0e+07 )
        _cd_ops=( 5.0e-8 5.0e-8 5.0e-8 )
    fi

    for m in "${!_size_target[@]}"
    do
        python kws_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log_supernet_val/srch_kws_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log
    done
fi

if [[ $_benchmark == "vww" ]]; then

    echo "vww benchmark"
    _epochs=70
    if [[ $_model == "PIT" ]]; then
        _size_target=( 1.30e+04 1.30e+04 1.30e+04 2.7e+04 2.7e+04 2.7e+04 5.3e+04 5.3e+04 5.3e+04 1.07e+05 1.07e+05 1.07e+05 )
        _cd_size=( 2.3e-05 2.3e-05 2.3e-05 2.5e-05 2.5e-05 2.5e-05 2.95e-05 2.95e-05 2.95e-05 4.4e-05 4.4e-05 4.4e-05 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _size_target=( 5.27e+04 5.27e+04 5.27e+04 1.06e+05 1.06e+05 1.06e+05 )
        _cd_size=( 6.21e-04 6.21e-04 6.21e-04 9.30e-04 9.30e-04 9.30e-04 )
    fi

    
    if [[ $_model == "PIT" ]]; then
        _latency_target=( 6.98e+05 1.40e+06 2.09e+06 8.56e+05 1.71e+06 2.57e+06 8.97e+05 1.79e+06 2.69e+06 8.85e+05 1.77e+06 2.65e+06 )
        _cd_ops=( 6.92e-06 7.71e-06 8.71e-06 7.08.5e-06 8.13e-06 9.55e-06 7.13e-06 8.25e-06 9.79e-06 7.12e-06 8.22e-06 9.72e-06 )
    fi
    if [[ $_model == "Supernet" ]]; then
        _latency_target=( 6.27e+05 1.34e+06 2.02e+06 1.41e+06 2.83e+06 4.24e+06 )
        _cd_ops=( 1.47e-05 1.63e-05 1.83e-05 1.65e-05 2.15e-05 3.08e-05 )
    fi

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