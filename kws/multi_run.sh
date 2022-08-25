#!/usr/bin/env bash

#size_target=( 1.8e+04 1.2e+04 6.0e+03 )
#cd_size=( 1.0e-05 5.4e-06 3.7e-06 )
#size_target=( 1.5e+05 )
#size_target=( 1.2e+04 )
#cd_size=( 5.4e-06 )
size_target=( 6.0e+03 )
cd_size=( 3.7e-06 )
#cd_ops=( 0.0e+00 1.0e-07 2.5e-07 5.0e-07 7.5e-07 5.0e-06 )
#cd_ops=( 0.0e+00 5.0e-08 1.0e-07 5.0e-07 1.0e-06 5.0e-06 )
#cd_ops=( 1.0e-08 1.0e-07 1.0e-06 )
#cd_ops=( 0.0e+00 5.0e-08 )
cd_ops=( 5.0e-09 )
#cd_ops=( 1.0e-08 5.0e-08 )

for i in "${!size_target[@]}"
do
    for cdo in "${cd_ops[@]}"
    do
        echo "Size-Target = ${size_target[i]}, Cd-Size = ${cd_size[i]},Cd-Ops = ${cdo}"
        source run.sh ${size_target[i]} ${cdo} ${cd_size[i]} search ft
    done
done