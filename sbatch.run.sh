#!/bin/bash
#SBATCH --mem=64G  # Memory
#SBATCH --constraint=vram48 #GPU size
#SBATCH -p gpu  # Partition
#SBATCH -G 1 # Number of GPUs
#SBATCH --qos=long
#SBATCH -t 96:00:00  # Job time limit
#SBATCH -o slurm_out/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
hostname

source activate /home/hheickal_umass_edu/.conda/envs/NLP_project
python main_cer.py exp_name='cerd' reconstruction_edit_flag=false lambda_contrastive=1 lambda_reconstruction=0 lambda_regularization=0


