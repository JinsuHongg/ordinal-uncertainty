#!/usr/bin/env bash
#SBATCH --job-name=utkface_prospective_smoke
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --output=logs/utkface_prospective_smoke/smoke_%j.out
#SBATCH --error=logs/utkface_prospective_smoke/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocqr
mkdir -p logs/utkface_prospective_smoke

echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits

python - <<'PY'
import importlib.util
import json
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "utkface_prospective", "scripts/run_utkface_prospective_replication.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

data_root = Path("/scratch/users/jhong36/data/utkface/extracted/UTKFace")
manifest = Path("/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("frozen smoke requires exactly one scheduler-visible CUDA device")

device = torch.device("cuda:0")
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "total_vram": torch.cuda.get_device_properties(0).total_memory,
}))

for objective in ("ce", "rps"):
    runner.set_seed(1)
    splits = runner.records(manifest, data_root)
    batches = runner.loader(
        splits["train"], data_root, train=True,
        batch=runner.BACKBONE_BATCH, shuffle=True,
    )
    images, labels, _ = next(iter(batches))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total = torch.cuda.mem_get_info(device)
    model = runner.make_resnet18(runner.K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    try:
        logits = model(images.to(device, non_blocking=True))
        value = runner.loss(objective, logits, labels.to(device, non_blocking=True))
        if not torch.isfinite(value):
            raise FloatingPointError(f"non-finite {objective} smoke loss")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        print(json.dumps({
            "objective": objective,
            "loss": float(value.detach().cpu()),
            "forward": True,
            "backward": True,
            "optimizer_step": True,
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
            "free_before": free_before,
            "free_after": free_after,
            "total_vram": total,
            "oom": False,
        }))
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"objective": objective, "oom": True}))
        raise
    finally:
        del model, optimizer, images, labels
        torch.cuda.empty_cache()
PY
