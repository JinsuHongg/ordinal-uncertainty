#!/usr/bin/env bash
#SBATCH --job-name=solar_ac_replication
#SBATCH --partition=qGPU24
#SBATCH --array=0-7
#SBATCH --output=logs/solar_ac_replication/ac_%A_%a.out
#SBATCH --error=logs/solar_ac_replication/ac_%A_%a.err
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
  0) OBJECTIVE=ce;  SEED=1 ;;
  1) OBJECTIVE=ce;  SEED=2 ;;
  2) OBJECTIVE=ce;  SEED=3 ;;
  3) OBJECTIVE=ce;  SEED=4 ;;
  4) OBJECTIVE=rps; SEED=1 ;;
  5) OBJECTIVE=rps; SEED=2 ;;
  6) OBJECTIVE=rps; SEED=3 ;;
  7) OBJECTIVE=rps; SEED=4 ;;
  *) echo "invalid array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

CHECKPOINT="outputs/solar/mechanism_replication/backbones/${OBJECTIVE}/seed_${SEED}/selected_checkpoint.pt"
OUT="outputs/mechanism_replication/ac/solar/${OBJECTIVE}/seed_${SEED}"
STATS="outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json"
test -f "$CHECKPOINT" || { echo "missing checkpoint $CHECKPOINT" >&2; exit 3; }
test -f "$STATS" || { echo "missing normalization artifact $STATS" >&2; exit 3; }
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p logs/solar_ac_replication

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} array_task=${SLURM_ARRAY_TASK_ID} cwd=$PWD"
echo "dataset=solar objective=$OBJECTIVE seed=$SEED checkpoint=$CHECKPOINT output=$OUT"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
nvidia-smi --query-gpu=name --format=csv,noheader
python - <<'PY'
import torch
print({'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda})
if not torch.cuda.is_available():
    raise SystemExit('CUDA is mandatory; refusing CPU fallback')
print({'gpu': torch.cuda.get_device_name(0)})
PY

python scripts/run_ac_mechanism_replication.py \
  --dataset solar --objective "$OBJECTIVE" --seed "$SEED" \
  --checkpoint "$CHECKPOINT" --out "$OUT" \
  --solar-stats "$STATS" --extract-batch 128 --workers 4 --device cuda:0
