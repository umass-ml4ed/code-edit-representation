#!/bin/bash
#SBATCH -G 1
#SBATCH -p gpu  # Partition
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o slurm-baseline-%j.out  # %j = job ID
#SBATCH --mail-type=END,FAIL
hostname

source activate /home/hheickal_umass_edu/.conda/envs/NLP_project
python openai_baselines_edit_descriptions.py
