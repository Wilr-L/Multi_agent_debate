#!/bin/bash
#SBATCH -J python
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:4 #根据需要设置，只使用CPU此行省略

source activate /scratch/users/k25159491/.conda/envs/embodi2
module load cuda
python /scratch/users/k25159491/WORK/Multi_agent_debate/Three_debate/evaluate_viki_l2_localmodel.py \
 --model-path /scratch/prj/inf_du/k25159491/Model/Qwen2.5-VL-32B-Instruct \
 --max-model-len 12864  --gpu-mem-util 0.85  --trust-remote-code --tensor-parallel-size 4