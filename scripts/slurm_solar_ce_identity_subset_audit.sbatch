#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_identity
#SBATCH --partition=qGPU24
#SBATCH --nodelist=acidsgcn013
#SBATCH --output=logs/solar_ce_identity_audit_%j.out
#SBATCH --error=logs/solar_ce_identity_audit_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
python scripts/audit_solar_ce_identity_subset.py
