#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-user=mkairov@hse.ru
#SBATCH --mail-type=ALL
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --time=0:15:00

module load Python/Anaconda_v11.2021
source deactivate
source activate meco_env

python3 Classifiers/cmd_run_classifiers.py -l du_fix -i "/home/mkairov/L2_MECO/Datasets/DataToUse/" -t