#!/usr/bin/env bash

benchmark=$1
gpu=$2
l=( "const" "increasing" )
loss=( "abs" "max" )
elements=( "mem" "mem+lat" )
if [[ "$1" == "icl" ]]; then
    size_target=( 2.0e+04 4.0e+04 6.0e+04 )
    cd_size=( 1.3e-05 5.9e-06 3.8e-06 )
    latency_target=(2.0e+06 6.0e+06 1.0e+07 )
    cd_ops=( 1.e-7 5e-9 1e-10 )
    if [[ $gpu == "0" ]]; then
        echo "Run on GPU 0"
        export CUDA_VISIBLE_DEVICES=0
        l="const"
        loss="abs" 
    fi
    if [[ $gpu == "1" ]]; then
        echo "Run on GPU 1"
        export CUDA_VISIBLE_DEVICES=1
        l="increasing"
        loss="abs" 
    fi
    if [[ $gpu == "2" ]]; then
        echo "Run on GPU 2"
        export CUDA_VISIBLE_DEVICES=2
        l="const"
        loss="max" 
    fi
    if [[ $gpu == "3" ]]; then
        echo "Run on GPU 3"
        export CUDA_VISIBLE_DEVICES=3
        l="increasing"
        loss="max" 
    fi
    for k in "${!elements[@]}"
    do
        for m in "${!size_target[@]}"
        do
            for n in "${!latency_target[@]}"
            do
                source run_flexnas.sh ${benchmark} ${cd_size[m]} ${size_target[m]} ${latency_target[n]} ${elements[k]} ${loss} ${cd_ops[n]} ${l}
            done
        done
    done
fi

if [[ "$1" == "vww" ]]; then
    # size_target=( 2.0e+04 4.0e+04 6.0e+04 )
    cd_size=( 5.3e-06 2.8e-06 1.9e-06 1.7e-06 )
    # latency_target=(2.0e+06 6.0e+06 1.0e+07 )
    cd_ops=( 5.3e-06 2.8e-06 1.9e-06 1.7e-06 )
    for i in "${!l[@]}"
    do
        for j in "${!loss[@]}"
        do
            for k in "${!elements[@]}"
            do
                for m in "${!size_target[@]}"
                do
                    for n in "${!latency_target[@]}"
                    do
                        source m100_offload.sbatch ${cd_size[m]} ${size_target[m]} ${latency_target[n]} ${elements[k]} ${loss[j]} ${cd_ops[n]} ${l[i]} ${benchmark}
                    done
                done
            done
        done
    done
fi

if [[ "$1" == "kws" ]]; then
    # size_target=( 2.0e+04 4.0e+04 6.0e+04 )
    cd_size=( 1.0e-05 5.4e-06 3.7e-06 )
    # latency_target=(2.0e+06 6.0e+06 1.0e+07 )
    cd_ops=( 1.0e-05 5.4e-06 3.7e-06 )
    for i in "${!l[@]}"
    do
        for j in "${!loss[@]}"
        do
            for k in "${!elements[@]}"
            do
                for m in "${!size_target[@]}"
                do
                    for n in "${!latency_target[@]}"
                    do
                        source m100_offload.sbatch ${cd_size[m]} ${size_target[m]} ${latency_target[n]} ${elements[k]} ${loss[j]} ${cd_ops[n]} ${l[i]} ${benchmark}
                    done
                done
            done
        done
    done
fi