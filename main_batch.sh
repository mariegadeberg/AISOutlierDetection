#!/bin/sh
# embedded options to bsub - start with #BSUB
# -- Name of the job ---
#BSUB -J vrnn_100
# -- specify queue --
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 5GB of memory per core/slot --
#BSUB -R "rusage[mem=12GB]"
### -- specify that we want the job to get killed if it exceeds 12 GB per core/slot -- 
#BSUB -M 48GB
### -- set walltime limit: hh:mm -- Maximum of 24 hours --
#BSUB -W 24:00 

### -- user email address --
#BSUB -u s153382@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
### -- end of LSF options --

# here follow the commands you want to execute

# Unload already installed software
module unload cuda
module unload cudnn

# load modules
module load cuda/10.2
module load cudnn/v7.6.5.32-prod-cuda-10.2

# run program
python AISOutlierDetection/train.py --num_epoch 10
