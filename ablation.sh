#! /bin/bash

models=$1
type=$2

for model in $models; do
    for factor in $(seq -5 1 5); do
        for percent in $(seq 1 1 5); do
            python ablation_$type.py \
                --model $model \
                --percent $percent \
                --factor $factor
        done
    done
done