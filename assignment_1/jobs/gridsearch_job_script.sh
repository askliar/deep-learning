#!/bin/bash
declare -a learning_rates=("0.001" "0.005" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.0001" "0.00001" )
declare -a batch_sizes=("50" "100" "500" "1000" "2000" "4000")
declare -a weight_decays=("0.01" "0.001" "0.005" "0.0001" "0.0005" )
declare -a dnn_hidden_units=("2048 1024 512 100")
declare -a dropouts=("0.0 0.0 0.0 0.0")
declare -a optimizers=("SGD" "Adam" "Adagrad")
declare -a momentums=("0.9") 

for lr in "${learning_rates[@]}"
do
	for batch_size in "${batch_sizes[@]}"
	do
		for weight_decay in "${weight_decays[@]}"
		do
			for i in {0..12}
			do
				for optimizer in "${optimizers[@]}"
				do
					for momentum in "${momentums[@]}"
					do
						temp=""$lr"_"$batch_size"_"$weight_decay"_"${dnn_hidden_units[i]}"_"$optimizer"_"${dropouts[i]}"_"$momentum""
						echo "$temp"
						qsub -v NAME="$temp",LEARNING_RATE="$lr",BATCH_SIZE="$batch_size",WEIGHT_DECAY="$weight_decay",OPTIMIZER="$optimizer",DNN_HIDDEN_UNITS="${dnn_hidden_units[i]}",MOMENTUM="$momentum",DROPOUTS="${dropouts[i]}" single_job_script.sh
					done
				done
			done
		done
	done
done
