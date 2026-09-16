#!/usr/bin/env bash
#SBATCH --job-name=solar_frozen_features
#SBATCH --partition=qGPU24
#SBATCH --array=0-7
#SBATCH --output=logs/solar_natural_direction/features_%A_%a.out
#SBATCH --error=logs/solar_natural_direction/features_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

CHECKPOINT="outputs/solar/mechanism_replication/backbones/${OBJECTIVE}/seed_${SEED}/selected_checkpoint.pt"
STATS="outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json"
OUT="outputs/mechanism_replication/features/solar/${OBJECTIVE}/seed_${SEED}"
test -f "$CHECKPOINT" && test -f "$STATS" && test ! -e "$OUT"
mkdir -p logs/solar_natural_direction

echo "job=${SLURM_JOB_ID:-unset} task=${SLURM_ARRAY_TASK_ID} objective=$OBJECTIVE seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/export_solar_confirmatory_frozen_features.py \
  --objective "$OBJECTIVE" --seed "$SEED" --checkpoint "$CHECKPOINT" \
  --stats "$STATS" --batch-size 128 --workers 4 --device cuda:0
