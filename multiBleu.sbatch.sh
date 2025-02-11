#!/bin/bash
#SBATCH --mem=64G  # Memory
#SBATCH --constraint=vram48 #GPU size
#SBATCH -p gpu  # Partition
#SBATCH -G 1 # Number of GPUs
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o slurm_out/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END,FAIL
hostname

module load conda/latest
source activate /home/hheickal_umass_edu/.conda/envs/NLP_project
python main_exp_multi_step_cerd_gen_code_decoder.py

