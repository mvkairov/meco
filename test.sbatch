#!/bin/bash
#SBATCH --job-name=test
#SBATCH --error=runs/test-%j.err
#SBATCH --output=runs/test-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --time=0:45:00

module load Python/Anaconda_v11.2021
source deactivate
conda create -n test_env python=3.9
source activate test_env
pip3 install pandas numpy scikit-learn requests

srun python Classifiers/cmd_run_classifiers.py -l ru_fix -t