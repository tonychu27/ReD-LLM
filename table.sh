#! /bin/bash

models=$1
attacks=$2

for model in $models; do
    python3 evaluate.py \
        --model $model \
        --attack $attacks \
        --detection_factor 1.0 \
        --refusal_factor 1.0 \
        --save_path table.csv
done