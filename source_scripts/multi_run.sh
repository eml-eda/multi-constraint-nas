#!/usr/bin/env bash

benchmark=$1
l=( "const" "increasing" )
loss=( "abs" "max" )
elements=( "mem" "mem+lat" )

if [[ "$1" == "icl" ]]; then
    size_target=( 2.0e+04 4.0e+04 6.0e+04 )
    cd_size=( 1.3e-05 5.9e-06 3.8e-06 )
    latency_target=(2.0e+06 6.0e+06 1.0e+07 )
    cd_ops=( 1.3e-5 5.9e-6 3.8e-6 )
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