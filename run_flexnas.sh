#!/usr/bin/env bash

size_target=2.0e+04 
cd_ops=0.0e+00

epochs=50
benchmark=$1
if [[ "$1" == "icl" ]]; then
cd_size=5e-4
python icl_training.py --cd-size ${cd_size} --size-target ${size_target} --cd-ops ${cd_ops} --epochs ${epochs} | tee log/srch_icl_${size_target}_${cd_ops}.log
fi
if [[ "$1" == "vww" ]]; then
cd_size=1.67e-06
python vww_training.py --cd-size ${cd_size} --size-target ${size_target} --cd-ops ${cd_ops} --epochs ${epochs} | tee log/srch_vww_${size_target}_${cd_ops}.log
fi
if [[ "$1" == "kws" ]]; then
cd_size=3.7e-06
python kws_training.py --cd-size ${cd_size} --size-target ${size_target} --cd-ops ${cd_ops} --epochs ${epochs} | tee log/srch_kws_${size_target}_${cd_ops}.log
fi
if [[ "$1" == "amd" ]]; then
cd_size=5e-4
python amd_training.py --cd-size ${cd_size} --size-target ${size_target} --cd-ops ${cd_ops} --epochs ${epochs} | tee log/srch_amd_${size_target}_${cd_ops}.log
fi