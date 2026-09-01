#!/usr/bin/env bash
#SBATCH --job-name=solar_ord_3ch_rps_s0
#SBATCH --output=logs/phase3_7a_3ch/%x_%j.out
#SBATCH --error=logs/phase3_7a_3ch/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
mkdir -p logs/phase3_7a_3ch outputs/solar/phase3_7a_3ch/rps
python scripts/phase3_7a_solar_3ch.py train --method rps --stats outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json --out outputs/solar/phase3_7a_3ch/rps/seed_0
