#!/usr/bin/env bash

size_target=2.0e+04 
cd_ops=0.0e+00
cd_size=5e-4

python icl_training.py --cd-size ${cd_size} --size-target ${size_target} \
        --cd-ops ${cd_ops} --epochs 10 | tee log/srch_icl_${size_target}_${cd_ops}.log