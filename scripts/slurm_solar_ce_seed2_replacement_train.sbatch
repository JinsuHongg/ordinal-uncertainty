#!/usr/bin/env bash
#SBATCH --job-name=solar_ce_s2_replacement_train
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --output=logs/solar_ce_seed2_replacement_train_%j.out
#SBATCH --error=logs/solar_ce_seed2_replacement_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

OUT="outputs/solar/mechanism_replication/replacement_seed2/ce/backbone"
test ! -e "$OUT" || { echo "replacement output already exists: $OUT" >&2; exit 3; }
PREFLIGHT="outputs/solar/mechanism_replication/replacement_seed2/ce/preflight_${SLURM_JOB_ID:-unset}"
export PREFLIGHT
mkdir -p logs "$PREFLIGHT"

echo "hostname=$(hostname) job=${SLURM_JOB_ID:-unset} partition=${SLURM_JOB_PARTITION:-unset} account=${SLURM_JOB_ACCOUNT:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python - <<'PY'
import json, os, platform, subprocess
from pathlib import Path
import torch, torchvision
if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f"required V100 unavailable: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'}")
expected = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "cuda": "12.4", "cudnn": 90100}
actual = {"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
if actual != expected:
    raise SystemExit(f"replacement environment mismatch: {actual} != {expected}")
record = {"hostname": subprocess.check_output(["hostname"], text=True).strip(), "job_id": os.getenv("SLURM_JOB_ID"), "partition": os.getenv("SLURM_JOB_PARTITION"), "account": os.getenv("SLURM_JOB_ACCOUNT"), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0), "device_properties": str(torch.cuda.get_device_properties(0)), "python": platform.python_version(), **actual, "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "matmul_precision": torch.get_float32_matmul_precision(), "autocast": False, "protocol_sha256": subprocess.check_output(["sha256sum", "docs/research/solar_ce_seed2_replacement_protocol.md"], text=True).split()[0]}
Path(os.environ["PREFLIGHT"]).joinpath("environment.json").write_text(json.dumps(record, indent=2)+"\n")
PY
python scripts/train_solar_replication_backbone.py \
  --method ce --seed 2 --out "$OUT" \
  --stats outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json \
  --workers 8
mv "$PREFLIGHT/environment.json" "$OUT/environment.json"
rmdir "$PREFLIGHT"
