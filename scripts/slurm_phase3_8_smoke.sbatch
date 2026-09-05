#!/usr/bin/env bash
#SBATCH --job-name=solar_p38_smoke
#SBATCH --output=logs/phase3_8/smoke_%j.out
#SBATCH --error=logs/phase3_8/smoke_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
OUT=outputs/solar/phase3_8_shrinkage_confirmation/smoke
test ! -e "$OUT" || { echo "Refusing to overwrite existing $OUT" >&2; exit 3; }
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unknown} python=$(command -v python) env=${CONDA_DEFAULT_ENV:-unknown}"
nvidia-smi || true
echo "git=$(git rev-parse HEAD 2>/dev/null || echo unavailable) cwd=$PWD config=scripts/phase3_8_solar_confirmation.py dataset=/scratch/users/jhong36/data/surya-bench-224.zarr output=$OUT seed=0"
python scripts/phase3_8_solar_confirmation.py smoke --method ce --stats outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json --out "$OUT/ce_seed_0"
python scripts/phase3_8_solar_confirmation.py smoke --method rps --stats outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json --out "$OUT/rps_seed_0"
