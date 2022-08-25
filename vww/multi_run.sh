#!/usr/bin/env bash

#size_target=( 1.5e+05 1.0e+05 5.0e+04 2.5e+04 )
#cd_size=( 5.3e-06 2.8e-06 1.9e-06 1.7e-06 )

#size_target=( 1.56e+05 )
#cd_size=( 5.86e-06 )

#size_target=( 1.04e+05 )
#cd_size=( 2.93e-06 )

#size_target=( 5.2e+04 )
#cd_size=( 1.95e-06 )

size_target=( 2.6e+04 )
cd_size=( 1.67e-06 )

#size_target=( 1.30e+04 )
#cd_size=( 1.56e-06 )

#cd_ops=( 0.0e+00 1.0e-09 5.0e-09 1.0e-08 )
cd_ops=( 2.50e-11 5.00e-11 )
#cd_ops=( 2.5e-10 5.0e-10 )

for i in "${!size_target[@]}"
do
    for cdo in "${cd_ops[@]}"
    do
        echo "Size-Target = ${size_target[i]}, Cd-Size = ${cd_size[i]}, Cd-Ops = ${cdo}"
        source run.sh ${size_target[i]} ${cdo} ${cd_size[i]} search ft
    done
done
