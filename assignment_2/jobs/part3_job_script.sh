#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=10:00:00

cd $HOME/uvadlc_practicals_2018/assignment_2/part3

echo "Job $PBS_JOBID started at `date`; text generation job is started" | mail $USER -s "Job $PBS_JOBID"

python train.py --txt_file books/book_of_jungle.txt --device 'cuda:0' --temperature "$TEMPERATURE" --sampling "$SAMPLING"
python train.py --txt_file books/Brodskiy.txt --device 'cuda:1' --temperature "$TEMPERATURE" --sampling "$SAMPLING"

echo "Job $PBS_JOBID ended at `date`; text generation job is finished" | mail $USER -s "Job $PBS_JOBID"