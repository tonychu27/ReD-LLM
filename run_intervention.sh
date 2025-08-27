#! /bin/bash

models=$1
attack=$2

for model in $models; do
    for detection_factor in $(seq 0 0.5 5); do
        for refusal_factor in $(seq 0 0.5 5); do
            python evaluate.py \
                --model $model \
                --attack $attack \
                --detection_factor $detection_factor \
                --refusal_factor $refusal_factor \
                --percent 3.0 \
                --save_path attack_result_$attack.csv
        done
    done
done 