#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_recipe_audit
#SBATCH --partition=qGPU24
#SBATCH --output=logs/solar_ce_historical_recipe_%j.out
#SBATCH --error=logs/solar_ce_historical_recipe_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
echo "job=${SLURM_JOB_ID:-unset} audit=solar_ce_historical_recipe"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
python scripts/audit_solar_ce_historical_recipe.py
