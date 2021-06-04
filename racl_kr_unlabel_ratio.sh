#!/bin/sh
#SBATCH -J Serial_gpu_job
#SBATCH -p ivy_v100_1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:2
#SBATCH --comment tensorflow

srun CUDA_VISIBLE_DEVICES=0 python ./scripts/RACL_self/tran_racl_kr.py --task dbsa --unlabel_ratio 0.5 --aug_num 0 --save0 0.5
srun python ./scripts/RACL_self/train_racl_kr.py --task dbsa --unlabel_ratio 1 --aug_num 0 --save0 1 
srun python ./scripts/RACL_self/train_racl_kr.py --task dbsa --unlabel_ratio 2 --aug_num 0 --save0 2  
srun python ./scripts/RACL_self/train_racl_kr.py --task dbsa --unlabel_ratio 4 --aug_num 0 --save0 4  
srun python ./scripts/RACL_self/train_racl_kr.py --task dbsa --unlabel_ratio 8 --aug_num 0 --save0 8  