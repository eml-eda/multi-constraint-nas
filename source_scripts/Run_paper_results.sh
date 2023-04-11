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
    _size_target=( 1.95e+04 3.9e+04 5.85e+04 )
    _cd_size=( 2.4e-05 3.6e-05 7.2e-05 )
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target_20=( 3.3e+06 4.15e+06 5.5e+06 8.3e+06 )
    _latency_target_40=( 3.3e+06 4.15e+06 5.5e+06 8.3e+06 )
    _latency_target_60=( 3.3e+06 4.15e+06 5.5e+06 8.3e+06 )
    _latency_target=( 3.3e+06 4.15e+06 5.5e+06 8.3e+06 3.3e+06 4.15e+06 5.5e+06 8.3e+06 3.3e+06 4.15e+06 5.5e+06 8.3e+06 )
    _cd_ops=( 5.0e-5 4.1e-5 3.3e-5 2.0e-5 5.0e-5 4.1e-5 3.3e-5 2.0e-5 5.0e-5 4.1e-5 3.3e-5 2.0e-5 )

    for m in "${!_size_target[@]}"
    do
        for n in "${!_latency_target_20[@]}"
        do
            # echo ${_cd_ops[m*4+n]} ${_latency_target[m*4+n]} ${_cd_size[m]} ${_size_target[m]}
            python icl_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m*4+n]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m*4+n]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_icl_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m*4+n]}_${_cd_ops[m*4+n]}_${_l}_${_elements}_${_loss}.log
        done
    done

fi

if [[ $_benchmark == "kws" ]]; then

    echo "kws benchmark"
    _epochs=36
    _size_target=( 1.0e+04 2.0e+04 3.0e+04 )
    _cd_size=( 6.6e-06 9.9e-06 2.0e-05 )
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target_20=( 0.7e+06 1.1e+06 1.5e+06 2.2e+06 )
    _latency_target_40=( 0.7e+06 1.1e+06 1.5e+06 2.2e+06 )
    _latency_target_60=( 0.7e+06 1.1e+06 1.5e+06 2.2e+06 )
    _latency_target=( 0.7e+06 1.1e+06 1.5e+06 2.2e+06 0.7e+06 1.1e+06 1.5e+06 2.2e+06 0.7e+06 1.1e+06 1.5e+06 2.2e+06 )
    _cd_ops=( 1.5e-4 7.0e-5 5.0e-5 4.1e-5 1.5e-4 7.0e-5 5.0e-5 4.1e-5 1.5e-4 7.0e-5 5.0e-5 4.1e-5 )

    for m in "${!_size_target[@]}"
    do
        for n in "${!_latency_target_20[@]}"
        do
            # echo ${_cd_ops[m*4+n]} ${_latency_target[m*4+n]} ${_cd_size[m]} ${_size_target[m]}
            python kws_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m*4+n]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m*4+n]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_kws_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m*4+n]}_${_cd_ops[m*4+n]}_${_l}_${_elements}_${_loss}.log
        done
    done

fi

if [[ $_benchmark == "vww" ]]; then

    echo "vww benchmark"
    _epochs=50
    _size_target=( 1.30e+04 2.7e+04 5.3e+04 1.07e+05 )
    _cd_size=( 2.3e-06 2.5e-06 2.95e-06 4.4e-06 )
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target_20=( 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 )
    _latency_target_40=( 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 )
    _latency_target_60=( 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 )
    _latency_target=( 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 1.1e+06 2.1e+06 3.3e+06 4.15e+06 5.5e+06 6.0e+06 )
    _cd_ops=( 1.5e-4 7.0e-5 5.0e-5 4.1e-5 3.3e-5 2.0e-5 1.5e-4 7.0e-5 5.0e-5 4.1e-5 3.3e-5 2.0e-5 1.5e-4 7.0e-5 5.0e-5 4.1e-5 3.3e-5 2.0e-5 1.5e-4 7.0e-5 5.0e-5 4.1e-5 3.3e-5 2.0e-5 )

    for m in "${!_size_target[@]}"
    do
        for n in "${!_latency_target_20[@]}"
        do
            # echo ${_cd_ops[m*4+n]} ${_latency_target[m*4+n]} ${_cd_size[m]} ${_size_target[m]}
            python vww_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m*6+n]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m*6+n]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_vww_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m*6+n]}_${_cd_ops[m*6+n]}_${_l}_${_elements}_${_loss}.log
        done
    done

fi

if [[ $_benchmark == "amd" ]]; then

    echo "amd benchmark"
    _epochs=100
    _size_target=( 0.33e+05 0.67e+05 1.34e+05 2.01e+05 )
    _cd_size=( 4.2e-06 4.98e-06 7.46e-06 1.5e-05 )
    ## We suppose 10 MACs/cycles, Fr. 100 MHz
    # 300, 240, 180, 120 FPS
    _latency_target=( 0.37e+05 0.75e+05 1.5e+05 2.25e+05 )
    _cd_ops=( 0.5e-03 1.0e-03 1.0e-04 1.0e-05 )

    for m in "${!_size_target[@]}"
    do
        # echo ${_cd_ops[m*4+n]} ${_latency_target[m*4+n]} ${_cd_size[m]} ${_size_target[m]}
        python amd_training.py --epochs ${_epochs} --cd-size ${_cd_size[m]} --cd-ops ${_cd_ops[m]} --size-target ${_size_target[m]} --latency-target ${_latency_target[m]} --loss_type ${_loss} --loss_elements ${_elements} --l ${_l} --model ${_model} --hardware ${_hardware} --gumbel ${_gumbel} | tee log/srch_amd_${ID}_${_gumbel}_${_model}_${_hardware}_${_size_target[m]}_${_cd_size[m]}_${_latency_target[m]}_${_cd_ops[m]}_${_l}_${_elements}_${_loss}.log

    done

fi