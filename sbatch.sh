#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=/scratch/users/%u/%j.err
#SBATCH --job-name=my_gpu_job
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=0-12:00

module load soft/anaconda3/config
source activate
conda activate embodi2

python evaluate_viki_l2_localmodel.py --model-path "/scratch/users/k25159491/WORK/Model/Qwen2.5-VL-3B-Instruct" \
    --tensor-parallel-size 2 --max-model-len 8192 --gpu-mem-util 0.90 \
    --trust-remote-code --limit 500
