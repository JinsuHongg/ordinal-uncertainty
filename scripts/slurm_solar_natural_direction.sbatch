#!/usr/bin/env bash
#SBATCH --job-name=solar_natural_direction
#SBATCH --partition=qGPU24
#SBATCH --array=0-7
#SBATCH --output=logs/solar_natural_direction/natural_%A_%a.out
#SBATCH --error=logs/solar_natural_direction/natural_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

case "${SLURM_ARRAY_TASK_ID:?missing array task id}" in
  0) OBJECTIVE=ce; SEED=1 ;;
  1) OBJECTIVE=ce; SEED=2 ;;
  2) OBJECTIVE=ce; SEED=3 ;;
  3) OBJECTIVE=ce; SEED=4 ;;
  4) OBJECTIVE=rps; SEED=1 ;;
  5) OBJECTIVE=rps; SEED=2 ;;
  6) OBJECTIVE=rps; SEED=3 ;;
  7) OBJECTIVE=rps; SEED=4 ;;
  *) exit 2 ;;
esac

FEATURE="outputs/mechanism_replication/features/solar/${OBJECTIVE}/seed_${SEED}/feature_manifest.json"
OUT="outputs/mechanism_replication/natural_direction/solar/${OBJECTIVE}/seed_${SEED}"
test -f "$FEATURE" && test ! -e "$OUT"
mkdir -p logs/solar_natural_direction

echo "job=${SLURM_JOB_ID:-unset} task=${SLURM_ARRAY_TASK_ID} objective=$OBJECTIVE seed=$SEED"
python scripts/run_solar_natural_direction.py --objective "$OBJECTIVE" --seed "$SEED" --device cuda:0
