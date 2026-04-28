#!/bin/bash
 
for i in $(seq 1 24); do
    echo "========================================="
    echo "Running dataset $i / 24"
    echo "========================================="
    python -u main.py \
        --dataset ~/dronesplat_dataset/${i}.mp4 \
        --config config/base.yaml \
        --save-as ~/MASt3R-SLAM/results/dronesplat/${i} \
		--no-viz
done