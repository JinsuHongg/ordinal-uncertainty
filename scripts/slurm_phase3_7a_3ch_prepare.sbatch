#!/usr/bin/env bash
#SBATCH --job-name=solar_ord_3ch_prepare
#SBATCH --output=logs/phase3_7a_3ch/%x_%j.out
#SBATCH --error=logs/phase3_7a_3ch/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
mkdir -p logs/phase3_7a_3ch outputs/solar/phase3_7a_3ch/normalization
python scripts/phase3_7a_solar_3ch.py prepare --out outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1
