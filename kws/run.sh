#!/usr/bin/env bash

size_target=$1
cd_ops=$2

arch="searchable_dscnn"
pretrained_model="pretrained_model/warmup.pth.tar"

if [[ "$3" == "search" ]]; then
    echo Search
    python search.py -a ${arch} \
        --cd-size 5e-5 --size-target ${size_target} \
        --cd-ops ${cd_ops} \
        --pretrained-model ${pretrained_model} | tee log/srch_${arch}_${size_target}_${cd_ops}.log
fi

if [[ "$4" == "ft" ]]; then
    echo Fine-Tune
    found_model="saved_models/srch_${arch}_target-${size_target}_cdops-${cd_ops}.pth.tar"
    python fine-tune.py -a ${arch} \
        --cd-size 5e-4 --size-target ${size_target} --cd-ops ${cd_ops} \
        --found-model ${found_model} | tee log/ft_${arch}_${size_target}_${cd_ops}.log
else
    echo From-Scratch
    python fine-tune.py -a ${arch} \
        --cd-size 5e-4 --size-target ${size_target} \
        --cd-ops ${cd_ops} | tee log/scrtch_${arch}_${size_target}_${cd_ops}.log
fi