#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_fp32_features
#SBATCH --partition=qGPU24
#SBATCH --array=1-4
#SBATCH --output=logs/solar_ce_strict_fp32/features_%A_%a.out
#SBATCH --error=logs/solar_ce_strict_fp32/features_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

SEED="${SLURM_ARRAY_TASK_ID:?missing array task id}"
CHECKPOINT="outputs/solar/mechanism_replication/backbones/ce/seed_${SEED}/selected_checkpoint.pt"
STATS="outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json"
OUT="outputs/mechanism_replication/features/solar/ce/seed_${SEED}"
test -f "$CHECKPOINT" && test -f "$STATS" && test ! -e "$OUT"
mkdir -p logs/solar_ce_strict_fp32
echo "job=${SLURM_JOB_ID:-unset} seed=${SEED} objective=ce strict_fp32=true"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "warning: nvidia-smi failed; rely on Slurm allocation and exporter runtime metadata"
python scripts/export_solar_confirmatory_frozen_features.py \
  --objective ce --seed "$SEED" --checkpoint "$CHECKPOINT" --stats "$STATS" \
  --batch-size 128 --workers 4 --device cuda:0 --strict-fp32
