#!/usr/bin/env bash
# Frozen UTKFace feature export array: submit once per objective.
# Example: sbatch --export=ALL,OBJECTIVE=ce scripts/slurm_utkface_prospective_feature_export.sbatch
#SBATCH --job-name=utkface_feature_export
#SBATCH --partition=qGPU24
#SBATCH --array=1-4
#SBATCH --output=logs/utkface_feature_export/%x_%A_%a.out
#SBATCH --error=logs/utkface_feature_export/%x_%A_%a.err
#SBATCH --time=04:00:00
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
CHECKPOINT="outputs/utkface/prospective_replication/backbones/${OBJECTIVE}/seed_${SEED}/best_checkpoint.pt"
OUT="outputs/utkface/prospective_replication/features/${OBJECTIVE}/seed_${SEED}"

test -d "$DATA_ROOT" || { echo "data root not found: $DATA_ROOT" >&2; exit 2; }
test -f "$MANIFEST" || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
test -f "$CHECKPOINT" || { echo "selected checkpoint not found: $CHECKPOINT" >&2; exit 2; }
test ! -e "$OUT" || { echo "refusing to overwrite existing output: $OUT" >&2; exit 3; }
mkdir -p logs/utkface_feature_export

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} array_task=${SEED} objective=${OBJECTIVE}"
echo "python=$(command -v python) conda_env=${CONDA_DEFAULT_ENV:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

COMMAND=(python scripts/run_utkface_prospective_replication.py export
  --objective "$OBJECTIVE" --seed "$SEED" --checkpoint "$CHECKPOINT" --out "$OUT"
  --data-root "$DATA_ROOT" --manifest "$MANIFEST" --device cuda:0)
printf 'command='
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"

OUT="$OUT" OBJECTIVE="$OBJECTIVE" SEED="$SEED" CHECKPOINT="$CHECKPOINT" \
SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURMD_NODENAME="${SLURMD_NODENAME:-$(hostname)}" \
python - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

out = Path(os.environ["OUT"])
metadata_path = out / "feature_export_metadata.json"
metadata = json.loads(metadata_path.read_text())
files = [out / "train_features.npz", out / "validation_features.npz", out / "test_features.npz", out / "A_original_head.pt"]
metadata["archive_sha256"] = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
}
metadata["execution"] = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": os.environ["SLURMD_NODENAME"],
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
    "command": ["python", "scripts/run_utkface_prospective_replication.py", "export", "--objective", os.environ["OBJECTIVE"], "--seed", os.environ["SEED"], "--checkpoint", os.environ["CHECKPOINT"], "--out", str(out), "--data-root", "/scratch/users/jhong36/data/utkface/extracted/UTKFace", "--manifest", "/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl", "--device", "cuda:0"],
    "environment": {"python": os.sys.version, "torch": __import__("torch").__version__, "cuda": __import__("torch").version.cuda},
}
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
PY
