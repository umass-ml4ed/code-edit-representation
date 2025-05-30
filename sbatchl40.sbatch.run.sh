#!/bin/bash
#SBATCH --mem=64G  # Memory
#SBATCH -G 1 # Number of GPUs
#SBATCH -p gpupod-l40s #superpod-a100 #gpupod-l40s # Partition
#SBATCH -q gpu-quota-12
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o slurm_out/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END,FAIL
hostname


source activate /home/hheickal_umass_edu/.conda/envs/NLP_project
python main_cer.py exp_name='cerd' epochs=100 lambda_reconstruction=3 lambda_regularization=2
