#!/bin/bash

models=$1
attack=$2
mode=$3

if [ -z "$models" ] || [ -z "$attack" ] || [ -z "$mode" ]; then
    echo "Usage: ./run_detection_heads.sh \"<model1 model2 ...>\" <attack> <mode>"
    echo "Example: ./run_detection_heads.sh \"llama3 mistral\" advllm scale"
    exit 1
fi


# for model in $models; do
#     for percent in $(seq 1 1 5); do
#         echo "Finding detection heads in $model with percent: $percent"
#         python find_detection_heads.py --target_model $model --percent $percent
#     done
# done

# for i in $(seq 1 1 5); do
    for model in $models; do
        # for percent in $(seq 1 1 5); do
            for factor in $(seq 0 0.5 5); do
                echo "Running $model with attack: $attack, factor: $factor, percent: $percent"
                python detection_head_intervention.py \
                    --target_model $model \
                    --attack $attack \
                    --mode $mode \
                    --percent 3.0 \
                    --factor $factor \
                    --save_path attack_result_$attack.csv
            done
        # done
    done
# done