#!/bin/bash
#SBATCH --job-name=wtmk_img_attack
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --mem=10G
#SBATCH -c 1
#SBATCH --ntasks 1
#SBATCH --output=attack100_bs100_delta2.out


python cv_attack.py