#!/usr/bin/env python3
"""A 64-row CE identity audit; never exports features or fits a model."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_7a_solar_3ch import EXPECTED, Solar, manifest, source_channels
from phase3_8_solar_confirmation import verify_stats
from ordinal_uncertainty.metrics.decision import bayes_decisions


OUT = Path("outputs/mechanism_replication/audits/solar_ce_identity_mismatch/subset_precision_audit.json")
ROOT = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
STATS = Path("outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json")
N = 64


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probabilities(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=1, keepdims=True)


def compare(logits: np.ndarray, archived: np.lib.npyio.NpzFile) -> dict[str, object]:
    target = archived["a_logits"][:N]
    probabilities_now = probabilities(logits)
    l1 = bayes_decisions(probabilities_now)["l1_bayes_decision"].astype(np.int64)
    delta = np.abs(logits - target)
    return {
        "logit_max_abs": float(delta.max()),
        "logit_mean_abs": float(delta.mean()),
        "logit_median_abs": float(np.median(delta)),
        "probability_max_abs": float(np.abs(probabilities_now - archived["a_probabilities"][:N]).max()),
        "predictive_mean_max_abs": float(np.abs(probabilities_now @ np.arange(5) - archived["a_predictive_mean"][:N]).max()),
        "changed_exact_l1": int((l1 != archived["a_l1"][:N]).sum()),
        "changed_sample_ids": archived["sample_ids"][:N][l1 != archived["a_l1"][:N]].astype(np.int64).tolist(),
    }


def forward(model: torch.nn.Module, dataset: Solar, batch: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(Subset(dataset, range(N)), batch_size=batch, shuffle=False, num_workers=0)
    logits, labels, ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for image, label, sample_id in loader:
            logits.append(model(image.to(device)).cpu())
            labels.append(label.cpu())
            ids.append(sample_id.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(ids).numpy()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for hardware/precision audit")
    source_channels(ROOT)
    stats = verify_stats(STATS)
    test_frame = manifest(Path(INDEX) / "test.csv", EXPECTED[2])
    dataset = Solar(test_frame, ROOT, (stats["mean"], stats["std"]), augment=False)
    device = torch.device("cuda:0")
    result: dict[str, object] = {
        "scope": "fixed first 64 archived-readout rows; no feature export, fitting, or training",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "stats_path": str(STATS), "stats_sha256": digest(STATS),
        "runs": {},
    }
    for seed in range(1, 5):
        # Recreate the failed A30 exporter default before each independent
        # comparison; the strict setting below must not leak across seeds.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        checkpoint = Path(f"outputs/solar/mechanism_replication/backbones/ce/seed_{seed}/selected_checkpoint.pt")
        source = Path(f"outputs/mechanism_replication/ac/solar/ce/seed_{seed}/per_sample_arrays.npz")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved["state_dict"]
        model = resnet18(weights=None); model.fc = torch.nn.Linear(512, 5)
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"non-strict CE state load: {loaded}")
        model = model.to(device).eval()
        with np.load(source, allow_pickle=False) as archived:
            default_logits, labels, ids = forward(model, dataset, 64, device)
            if not np.array_equal(ids, archived["sample_ids"][:N]) or not np.array_equal(labels, archived["labels"][:N]):
                raise RuntimeError("subset sample alignment failure")
            default = compare(default_logits, archived)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            strict_logits_64, _, _ = forward(model, dataset, 64, device)
            strict_logits_32, _, _ = forward(model, dataset, 32, device)
            strict = compare(strict_logits_64, archived)
            result["runs"][str(seed)] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "checkpoint_metadata": {k: saved.get(k) for k in ("method", "seed", "selected_epoch", "validation_loss")},
                "strict_state_load": True, "eval_mode": not model.training,
                "bn_running_buffers_present": all(k in state for k in ("bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked")),
                "default_a30": default, "strict_fp32_a30": strict,
                "strict_batch32_vs_64_max_abs": float(np.abs(strict_logits_32 - strict_logits_64).max()),
            }
        del model
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
