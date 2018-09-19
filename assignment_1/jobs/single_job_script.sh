#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=10:00:00

cd $HOME/uvadlc_practicals_2018/assignment_1/code

echo "Job $PBS_JOBID with spatial started at `date`; $DNN_HIDDEN_UNITS; $NAME is started" | mail $USER -s "Job $PBS_JOBID"

python train_mlp_pytorch.py --dnn_hidden_units "$DNN_HIDDEN_UNITS" --learning_rate "$LEARNING_RATE" --optimizer "$OPTIMIZER" --weight_decay "$WEIGHT_DECAY" --batch_size "$BATCH_SIZE" --dnn_dropouts "$DROPOUTS" --momentum "$MOMENTUM"

echo "Job $PBS_JOBID with spatial ended at `date`; "$NAME" is finished" | mail $USER -s "Job $PBS_JOBID"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.0" --learning_rate "0.01" --batch_size "512" --optimizer "SGD" --weight_decay "0.0005" --momentum "0.0"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.0" --learning_rate "0.01" --batch_size "512" --optimizer "SGD" --weight_decay "0.0005" --momentum "0.9"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.2" --learning_rate "0.01" --batch_size "512" --optimizer "SGD" --weight_decay "0.0005" --momentum "0.9"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512,100" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "SGD" --weight_decay "0.00005" --momentum "0.9"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,1024,512,100" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "SGD" --weight_decay "0.00005" --momentum "0.9"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "2048,1024,1024,512" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "SGD" --weight_decay "0.00005" --momentum "0.9"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "Adagrad" --weight_decay "0.00005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "Adagrad" --weight_decay "0.00005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.2" --learning_rate "0.001" --batch_size "512" --optimizer "Adagrad" --weight_decay "0.00005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512,100" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "Adagrad" --weight_decay "0.00005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,1024,512,100" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "Adagrad" --weight_decay "0.00005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "2048,1024,1024,512" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.001" --batch_size "512" --optimizer "Adagrad" --weight_decay "0.00005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.0" --learning_rate "0.00001" --batch_size "512" --optimizer "Adam" --weight_decay "0.000005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.0" --learning_rate "0.00001" --batch_size "512" --optimizer "Adam" --weight_decay "0.000005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512" --dnn_dropouts "0.2,0.2,0.2" --learning_rate "0.00001" --batch_size "512" --optimizer "Adam" --weight_decay "0.000005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,512,512,100" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.00001" --batch_size "512" --optimizer "Adam" --weight_decay "0.000005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "1024,1024,512,100" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.00001" --batch_size "512" --optimizer "Adam" --weight_decay "0.000005"

# qsub -qgpu -lwalltime=10:00:00 -lnodes=1 train_mlp_pytorch.py --dnn_hidden_units "2048,1024,1024,512" --dnn_dropouts "0.2,0.2,0.2,0.0" --learning_rate "0.00001" --batch_size "512" --optimizer "Adam" --weight_decay "0.000005"