#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=10:00:00

cd $HOME/uvadlc_practicals_2018/assignment_1/code

echo "Job $PBS_JOBID with spatial started at `date`; convnet is started" | mail $USER -s "Job $PBS_JOBID"

python train_convnet_pytorch.py

echo "Job $PBS_JOBID with spatial ended at `date`; convnet is finished" | mail $USER -s "Job $PBS_JOBID"