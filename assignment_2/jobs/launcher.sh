#!/bin/bash
declare -a models=("LSTM" "RNN")
declare -a sizes=("5" "6" "7" "8" "9" "10" "11" "12" "13")

for model in "${models[@]}"
do
	for size in "${sizes[@]}"
	do
        temp=""$model"_"$size""
        echo "$temp"
        qsub -v MODEL_TYPE="$model",INPUT_LENGTH="$size" single_job_script.sh
    done
done