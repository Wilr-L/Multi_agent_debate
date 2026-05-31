#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 #根据需要设置，只使用CPU此行省略
#SBATCH -x g01
 
source /apps/soft/anaconda3/bin/activate
conda activate embodi
python /scratch/users/k25159491/WORK/Multi_agent_debate/Three_debate/evaluate_viki_l2_localmodel.py \
 --model-path /scratch/prj/inf_du/k25159491/Model/Qwen2.5-VL-7B-Instruct \
 --max-model-len 12864  --gpu-mem-util 0.85  --trust-remote-code \
 --limit 300