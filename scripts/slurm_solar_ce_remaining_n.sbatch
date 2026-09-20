#!/usr/bin/env bash
# Solar CE auxiliary natural-direction follow-up: exactly seeds 1, 3, and 4.
# The runner itself fail-closes on Phase-A, archive, A/C identity, and C-contract
# checks before it can create an output directory.
#SBATCH --job-name=solar_ce_natural
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --nodelist=acidsgcn007
#SBATCH --gres=gpu:V100:1
#SBATCH --array=1,3,4
#SBATCH --output=logs/solar_ce_remaining_n/natural_%A_%a.out
#SBATCH --error=logs/solar_ce_remaining_n/natural_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

SEED="${SLURM_ARRAY_TASK_ID:?missing Solar CE seed}"
case "$SEED" in 1|3|4) ;; *) echo "unauthorized seed: $SEED" >&2; exit 2 ;; esac

FEATURE="outputs/mechanism_replication/features/solar/ce/seed_${SEED}/feature_manifest.json"
OUTPUT="outputs/mechanism_replication/natural_direction/solar/ce/seed_${SEED}"
test -f "$FEATURE"
test ! -e "$OUTPUT"
mkdir -p logs/solar_ce_remaining_n

echo "job=${SLURM_JOB_ID:-unset} task=${SLURM_ARRAY_TASK_ID} objective=ce seed=$SEED host=$(hostname) partition=${SLURM_JOB_PARTITION:-unset} account=${SLURM_JOB_ACCOUNT:-unset} visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0)})
if "V100" not in torch.cuda.get_device_name(0):
    raise RuntimeError("runtime GPU is not V100")
PY

python scripts/run_solar_natural_direction.py --objective ce --seed "$SEED" --device cuda:0
