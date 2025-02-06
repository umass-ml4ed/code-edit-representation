#!/bin/bash
#SBATCH --mem=64G  # Memory
#SBATCH -G 1 # Number of GPUs
#SBATCH -p gpupod-l40s #superpod-a100 # # Partition
#SBATCH -q gpu-quota-12
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o slurm_out/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END,FAIL
hostname

source activate /home/hheickal_umass_edu/.conda/envs/NLP_project
python main_exp_multi_step_cerd_gen_code_decoder.py


