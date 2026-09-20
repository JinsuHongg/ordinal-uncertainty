#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_s2_replacement_n
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --output=logs/solar_ce_seed2_replacement_n_%j.out
#SBATCH --error=logs/solar_ce_seed2_replacement_n_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python scripts/run_solar_ce_seed2_replacement_n.py
