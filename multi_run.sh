#!/usr/bin/env bash

size_target=( 2.6e+04 2.6e+04 2.6e+04 )
cd_size=5e-4

latency_target=2.0e+06
cd_ops=( 1.3e-5 5.9e-6 3.8e-6 )
l=( "const" "increasing" )
loss=( "abs" "max" )
elements=( "mem" "mem+lat" )

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
                    source m100_offload.sbatch ${cd_size} ${size_target[m]} ${latency_target[n]} ${elements[k]} ${loss[j]} ${cd_ops[n]} ${l[i]}
                done
            done
        done
    done
done