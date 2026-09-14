#!/usr/bin/env bash
#SBATCH --job-name=solar_rep_backbone
#SBATCH --partition=qGPU24
#SBATCH --array=0-7
#SBATCH --output=logs/solar_mechanism_replication/backbone_%A_%a.out
#SBATCH --error=logs/solar_mechanism_replication/backbone_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

case "${SLURM_ARRAY_TASK_ID:?missing array task id}" in
  0) METHOD=ce;  SEED=1 ;;
  1) METHOD=ce;  SEED=2 ;;
  2) METHOD=ce;  SEED=3 ;;
  3) METHOD=ce;  SEED=4 ;;
  4) METHOD=rps; SEED=1 ;;
  5) METHOD=rps; SEED=2 ;;
  6) METHOD=rps; SEED=3 ;;
  7) METHOD=rps; SEED=4 ;;
  *) echo "invalid array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
OUT="outputs/solar/mechanism_replication/backbones/${METHOD}/seed_${SEED}"
test ! -e "$OUT" || { echo "Refusing to overwrite existing condition $OUT" >&2; exit 3; }
mkdir -p logs/solar_mechanism_replication
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} array_task=${SLURM_ARRAY_TASK_ID} cwd=$PWD"
echo "method=$METHOD seed=$SEED output=$OUT python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
nvidia-smi --query-gpu=name --format=csv,noheader
python - <<'PY'
import torch
print({'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda})
if not torch.cuda.is_available():
    raise SystemExit('CUDA is mandatory; refusing CPU fallback')
print({'gpu': torch.cuda.get_device_name(0)})
PY
python scripts/train_solar_replication_backbone.py \
  --method "$METHOD" --seed "$SEED" --out "$OUT" \
  --stats outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json
