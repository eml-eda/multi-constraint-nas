#!/usr/bin/env bash

size_target_25=$1
cd_ops=$2

# Compute 75% and 50% size target (now hard-coded to be faster)
size_target_75=1.5e+05
size_target_50=1.0e+05

cd_size=5e-6
arch="searchable_mobilenetv1"
pretrained_model="pretrained_model/warmup.pth.tar"

echo "Search 75% for 10 epochs"
python search.py -a ${arch} --epochs 10 \
    --cd-size 5e-4 --size-target ${size_target_75} \
    --cd-ops ${cd_ops} \
    --pretrained-model ${pretrained_model} | tee log/srch_${arch}_${size_target_25}_${cd_ops}_75.log
pretrained_model="saved_models/srch_${arch}_target-${size_target_75}_cdops-${cd_ops}.pth.tar"

echo "Search 50% for 10 epochs"
python search.py -a ${arch} --epochs 10 \
    --cd-size 5e-4 --size-target ${size_target_50} \
    --cd-ops ${cd_ops} \
    --pretrained-model ${pretrained_model} | tee log/srch_${arch}_${size_target_25}_${cd_ops}_50.log
pretrained_model="saved_models/srch_${arch}_target-${size_target_50}_cdops-${cd_ops}.pth.tar"

echo "Search 25% for 50 epochs"
python search.py -a ${arch} --epochs 50 --early-stop 10 \
    --cd-size ${cd_size} --size-target ${size_target_25} \
    --cd-ops ${cd_ops} --lr 1e-4 \
    --pretrained-model ${pretrained_model} | tee log/srch_${arch}_${size_target_25}_${cd_ops}_25.log

echo Fine-Tune
found_model="saved_models/srch_${arch}_target-${size_target_25}_cdops-${cd_ops}.pth.tar"
python fine-tune.py -a ${arch} \
    --cd-size ${cd_size} --size-target ${size_target} --cd-ops ${cd_ops} \
    --found-model ${found_model} | tee log/ft_${arch}_${size_target}_${cd_ops}.log
