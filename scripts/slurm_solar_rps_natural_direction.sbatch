#!/usr/bin/env bash
#SBATCH --job-name=solar_rps_natural
#SBATCH --partition=qGPU24
#SBATCH --array=1-4
#SBATCH --output=logs/solar_natural_direction/rps_natural_%A_%a.out
#SBATCH --error=logs/solar_natural_direction/rps_natural_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

SEED="${SLURM_ARRAY_TASK_ID:?missing array task id}"
FEATURE="outputs/mechanism_replication/features/solar/rps/seed_${SEED}/feature_manifest.json"
OUT="outputs/mechanism_replication/natural_direction/solar/rps/seed_${SEED}"
test -f "$FEATURE" && test ! -e "$OUT"
mkdir -p logs/solar_natural_direction

echo "job=${SLURM_JOB_ID:-unset} seed=${SEED} objective=rps"
python scripts/run_solar_natural_direction.py --objective rps --seed "$SEED" --device cuda:0
