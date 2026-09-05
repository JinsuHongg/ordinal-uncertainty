#!/usr/bin/env bash
#SBATCH --job-name=solar_p39_rps_features
#SBATCH --output=logs/phase3_9/rps_features_%j.out
#SBATCH --error=logs/phase3_9/rps_features_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ocqr
OUT=outputs/solar/phase3_9_mechanism_audit/features/rps
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p logs/phase3_9
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unknown} env=${CONDA_DEFAULT_ENV:-unknown} cwd=$PWD"
nvidia-smi || true
python scripts/phase3_9_solar_mechanism_audit.py extract --method rps --out "$OUT"
