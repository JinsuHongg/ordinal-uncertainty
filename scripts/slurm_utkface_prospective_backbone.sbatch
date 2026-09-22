#!/usr/bin/env bash
# Frozen UTKFace prospective backbone array: submit once per objective.
# Example: sbatch --export=ALL,OBJECTIVE=ce scripts/slurm_utkface_prospective_backbone.sbatch
#SBATCH --job-name=utkface_prospective_backbone
#SBATCH --partition=qGPU24
#SBATCH --array=1-4
#SBATCH --output=logs/utkface_prospective_backbone/%x_%A_%a.out
#SBATCH --error=logs/utkface_prospective_backbone/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr

: "${OBJECTIVE:?submit with OBJECTIVE=ce or OBJECTIVE=rps}"
case "$OBJECTIVE" in
  ce|rps) ;;
  *) echo "invalid OBJECTIVE: $OBJECTIVE" >&2; exit 2 ;;
esac

DATA_ROOT=/scratch/users/jhong36/data/utkface/extracted/UTKFace
MANIFEST=/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl
SEED=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}
OUT="outputs/utkface/prospective_replication/backbones/${OBJECTIVE}/seed_${SEED}"

test -d "$DATA_ROOT" || { echo "data root not found: $DATA_ROOT" >&2; exit 2; }
test -f "$MANIFEST" || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
test ! -e "$OUT" || { echo "refusing to overwrite existing output: $OUT" >&2; exit 3; }
mkdir -p logs/utkface_prospective_backbone

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} array_task=${SEED} objective=${OBJECTIVE}"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is mandatory; refusing CPU fallback")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PY

COMMAND=(python scripts/run_utkface_prospective_replication.py backbone
  --objective "$OBJECTIVE" --seed "$SEED" --out "$OUT"
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --device cuda:0)
printf 'command='
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"

OUT="$OUT" OBJECTIVE="$OBJECTIVE" SEED="$SEED" DATA_ROOT="$DATA_ROOT" MANIFEST="$MANIFEST" \
SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURMD_NODENAME="${SLURMD_NODENAME:-$(hostname)}" \
python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

out = Path(os.environ["OUT"])
config = json.loads((out / "config.json").read_text())
checkpoint = out / "best_checkpoint.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
command = [
    "python", "scripts/run_utkface_prospective_replication.py", "backbone",
    "--objective", os.environ["OBJECTIVE"], "--seed", os.environ["SEED"],
    "--out", str(out), "--data-root", os.environ["DATA_ROOT"],
    "--manifest", os.environ["MANIFEST"], "--device", "cuda:0",
]
payload = {
    "loss": os.environ["OBJECTIVE"], "seed": int(os.environ["SEED"]),
    "job_id": os.environ["SLURM_JOB_ID"], "host": os.environ["SLURMD_NODENAME"],
    "gpu": gpu, "command": command, "data_root": os.environ["DATA_ROOT"],
    "manifest": os.environ["MANIFEST"], "manifest_sha256": config["manifest_sha256"],
    "environment": {"python": os.sys.version, "torch": __import__("torch").__version__, "cuda": __import__("torch").version.cuda},
    "selected_epoch": config["best_epoch"], "validation_metric": config["best_validation_score"],
    "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
    "completion_status": "complete", "retry_provenance": "initial submission",
}
(out / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
