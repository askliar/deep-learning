#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=10:00:00

cd $HOME/uvadlc_practicals_2018/assignment_2/part1

echo "Job $PBS_JOBID with spatial started at `date`; $MODEL_TYPE with $INPUT_LENGTH is started" | mail $USER -s "Job $PBS_JOBID"

python train.py --model_type "$MODEL_TYPE" --input_length "$INPUT_LENGTH"

echo "Job $PBS_JOBID with spatial ended at `date`; $MODEL_TYPE with $INPUT_LENGTH  is finished" | mail $USER -s "Job $PBS_JOBID"