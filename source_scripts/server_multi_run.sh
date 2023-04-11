#!/usr/bin/env bash

benchmark=$1
gpu=$2
hardware=$3
model=$4
l=( "const" "increasing" )
loss=( "abs" "max" )
elements=( "mem" "mem+lat" )
latency_target_None=( 2.0e+06 6.0e+06 1.0e+07 )
latency_target_Diana=( 1.0e+04 4.0e+04 7.0e+04 )
latency_target_GAP8=( 1.0e+05 4.0e+05 7.0e+05 )

if [[ "$1" == "icl" ]]; then
    size_target=( 2.0e+04 4.0e+04 6.0e+04 )
    cd_size=( 4.0e-05 3.6e-05 3.6e-05 )
    if [[ "$3" == "GAP8" ]]; then
        latency_target=( 1.0e+05 4.0e+05 7.0e+05 )
        if [[ "$4" == "PIT" ]]; then
            cd_ops=( 5.e-5 5e-8 5e-9 )
        fi
        if [[ "$4" == "Supernet" ]]; then
            cd_ops=( 5.e-8 5e-11 5e-12 )
        fi
    fi
    if [[ "$3" == "Diana" ]]; then
        latency_target=( 1.0e+04 4.0e+04 7.0e+04 )
        cd_ops=( 5.e-4 5e-7 5e-8 )
        if [[ "$4" == "PIT" ]]; then
            cd_ops=( 5.e-4 5e-7 5e-8 )
        fi
        if [[ "$4" == "Supernet" ]]; then
            cd_ops=( 5.e-7 5e-10 5e-11 )
        fi
    fi
    if [[ "$3" == "None" ]]; then
        latency_target=( 2.0e+06 6.0e+06 1.0e+07 )
        cd_ops=( 1.e-7 5e-9 1e-10 )
        if [[ "$4" == "PIT" ]]; then
            cd_ops=( 1.e-7 5e-9 1e-10 )
        fi
        if [[ "$4" == "Supernet" ]]; then
            cd_ops=( 1.e-10 5e-12 1e-13 )
        fi
    fi
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
                echo ${size_target[m]} ${latency_target[n]} ${elements[k]} 
                source run_flexnas.sh ${benchmark} ${cd_size[m]} ${size_target[m]} ${latency_target[n]} ${elements[k]} ${loss} ${cd_ops[n]} ${l} ${model} ${hardware}
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