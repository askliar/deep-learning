#!/bin/bash
declare -a sampling=("greedy" "random")
declare -a temperatures=("0.5" "1.0" "2.0")

for temperature in "${temperatures[@]}"
do
    temp="random_"$temperature""
    echo "$temp"
    qsub -v SAMPLING="random",TEMPERATURE="$temperature" part3_job_script.sh
done

qsub -v SAMPLING="greedy",TEMPERATURE="1.0" part3_job_script.sh
