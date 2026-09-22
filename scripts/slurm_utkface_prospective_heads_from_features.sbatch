#!/usr/bin/env bash
#SBATCH --job-name=utkface_acn_heads
#SBATCH --partition=qGPU24
#SBATCH --array=1-4
#SBATCH --output=logs/utkface_prospective_heads/%x_%A_%a.out
#SBATCH --error=logs/utkface_prospective_heads/%x_%A_%a.err
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
case "$OBJECTIVE" in ce|rps) ;; *) exit 2;; esac
SEED=${SLURM_ARRAY_TASK_ID:?}
OUT="outputs/utkface/prospective_replication/heads/${OBJECTIVE}/seed_${SEED}"
MANIFEST="outputs/utkface/prospective_replication/features/feature_export_manifest.json"
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p logs/utkface_prospective_heads
echo "job=${SLURM_JOB_ID:-} host=$(hostname) objective=$OBJECTIVE seed=$SEED cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python - <<'PY'
import torch
if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name(0): raise SystemExit('V100 CUDA allocation required')
print({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)})
PY
python scripts/run_utkface_prospective_heads_from_features.py --objective "$OBJECTIVE" --seed "$SEED" --feature-manifest "$MANIFEST" --out "$OUT" --device cuda:0
OUT="$OUT" SLURM_JOB_ID="${SLURM_JOB_ID:-}" python - <<'PY'
import hashlib,json,os
from pathlib import Path
p=Path(os.environ['OUT'])/'manifest.json'; d=json.loads(p.read_text()); d['execution']={'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename}; d['completion_status']='complete'; d['manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); p.write_text(json.dumps(d,indent=2)+'\n')
PY
