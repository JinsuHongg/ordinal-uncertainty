#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_s2_replacement_features
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --output=logs/solar_ce_seed2_replacement_features_%j.out
#SBATCH --error=logs/solar_ce_seed2_replacement_features_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unset} partition=${SLURM_JOB_PARTITION:-unset} account=${SLURM_JOB_ACCOUNT:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python scripts/export_solar_ce_seed2_replacement_features.py
