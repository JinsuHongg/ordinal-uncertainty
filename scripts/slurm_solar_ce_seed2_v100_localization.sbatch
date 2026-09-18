#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_s2_localize
#SBATCH --partition=qGPU24
#SBATCH --output=logs/solar_ce_seed2_v100_localization_%j.out
#SBATCH --error=logs/solar_ce_seed2_v100_localization_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
echo "job=${SLURM_JOB_ID:-unset} diagnostic=solar_ce_seed2_v100_localization"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/audit_solar_ce_seed2_v100_localization.py
