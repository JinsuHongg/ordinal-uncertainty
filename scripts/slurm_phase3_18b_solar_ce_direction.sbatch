#!/usr/bin/env bash
#SBATCH --job-name=solar_p318b_ce_direction
#SBATCH --partition=qGPU24
#SBATCH --output=logs/phase3_18b/solar_ce_direction_%j.out
#SBATCH --error=logs/phase3_18b/solar_ce_direction_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
OUT=outputs/solar/phase3_18b_ce_direction_robustness
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p logs/phase3_18b
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cwd=$PWD"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
nvidia-smi --query-gpu=name --format=csv,noheader
python - <<'PY'
import torch
print({'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda})
if not torch.cuda.is_available():
    raise SystemExit('CUDA is mandatory; refusing CPU fallback')
print({'gpu': torch.cuda.get_device_name(0)})
PY
python scripts/run_phase3_18b_solar_ce_direction.py
