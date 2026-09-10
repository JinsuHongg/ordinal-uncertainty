#!/usr/bin/env python3
"""Phase 3.18B: frozen Solar CE A/B/C direction robustness confirmation."""
from __future__ import annotations

import csv
import json
import os
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices
from phase3_7a_solar_3ch import report


ROOT = Path("outputs/solar/phase3_18b_ce_direction_robustness")
FEATURE_ROOT = Path("outputs/solar/phase3_9_mechanism_audit/features/ce")
CHECKPOINT = Path("outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/selected_checkpoint.pt")
BATCH_SIZE, EPOCHS = 64, 100


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def load_cache(split: str) -> dict[str, np.ndarray]:
    path = FEATURE_ROOT / f"{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "sample_ids", "logits"}
        if required - set(archive.files):
            raise ValueError(f"{path} missing {sorted(required - set(archive.files))}")
        data = {key: archive[key].copy() for key in archive.files}
    if data["features"].ndim != 2 or data["features"].shape[1] != 512:
        raise ValueError(f"unexpected frozen feature shape: {data['features'].shape}")
    if not np.isfinite(data["features"]).all() or not np.isfinite(data["logits"]).all():
        raise ValueError(f"non-finite cached values in {path}")
    if not np.isin(data["labels"], np.arange(5)).all() or len(np.unique(data["sample_ids"])) != len(data["sample_ids"]):
        raise ValueError(f"invalid labels or non-unique IDs in {path}")
    return data


def checkpoint_head() -> tuple[torch.Tensor, torch.Tensor]:
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved.get("model_state_dict", saved))
    weight, bias = state["fc.weight"].float(), state["fc.bias"].float()
    if weight.shape != (5, 512) or bias.shape != (5,):
        raise ValueError(f"unexpected CE head shapes {weight.shape}/{bias.shape}")
    return weight, bias


def fit(head: nn.Module, x: torch.Tensor, y: torch.Tensor, weight_decay: float) -> list[dict]:
    params = head.parameters() if isinstance(head, nn.Linear) else [head.direction]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=weight_decay)
    # Resetting this generator gives B and C the exact same balanced batches.
    generator = torch.Generator().manual_seed(10000)
    history = []
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for _ in range(int(np.ceil(len(y) / BATCH_SIZE))):
            index = balanced_batch_indices(y, BATCH_SIZE, generator).to(x.device)
            loss = torch.nn.functional.cross_entropy(head(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "balanced_ce": float(np.mean(losses))})
    return history


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.detach().cpu()[None], b.detach().cpu()[None]).item())


def x_geometry_subgroups(train: dict[str, np.ndarray], test: dict[str, np.ndarray], evaluations: Path) -> list[dict]:
    """Phase-3.9-compatible raw train-centroid descriptive stratification."""
    centroids = np.stack([train["features"][train["labels"] == k].mean(0) for k in range(5)])
    xmask = test["labels"] == 4
    nearest = ((test["features"][xmask, None, :] - centroids[None, :, :]) ** 2).sum(2).argmin(1)
    rows = []
    for group, groupmask in (("x_like_raw_nearest_x", nearest == 4), ("representation_inward_raw_nearest_interior", nearest != 4)):
        for condition in ("A_original_ce", "C_direction_only"):
            with np.load(evaluations / condition / "predictions.npz", allow_pickle=False) as p:
                l1 = p["l1"][xmask][groupmask]
            rows.append({"group": group, "condition": condition, "support": int(groupmask.sum()),
                         "l1_routing_0_to_4": "/".join(str(int((l1 == k).sum())) for k in range(5)),
                         "exact_x": int((l1 == 4).sum()), "mae": float(np.abs(4 - l1).mean())})
    return rows


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Phase 3.18B; refusing CPU fallback")
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = torch.device("cuda:0")
    print(json.dumps({"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                      "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(device),
                      "checkpoint": str(CHECKPOINT), "feature_cache": str(FEATURE_ROOT)}, indent=2), flush=True)
    train, validation, test = (load_cache(name) for name in ("train", "validation", "test"))
    identities = {n: set(d["sample_ids"].tolist()) for n, d in (("train", train), ("validation", validation), ("test", test))}
    overlaps = {"train_validation": len(identities["train"] & identities["validation"]), "train_test": len(identities["train"] & identities["test"]), "validation_test": len(identities["validation"] & identities["test"])}
    if any(overlaps.values()): raise ValueError(f"split ID overlap: {overlaps}")
    original_weight, original_bias = checkpoint_head()
    replay = test["features"] @ original_weight.numpy().T + original_bias.numpy()
    replay_error = float(np.abs(replay - test["logits"]).max())
    if replay_error > 2e-5: raise ValueError(f"cached-test logit replay error {replay_error}")
    ROOT.mkdir(parents=True)
    integrity = {"protocol": "Phase 3.18B frozen Solar CE A/B/C", "feature_source": str(FEATURE_ROOT), "checkpoint": str(CHECKPOINT), "feature_dimension": 512,
                 "counts": {n: int(len(d["labels"])) for n, d in (("train", train), ("validation", validation), ("test", test))},
                 "class_counts": {n: [int((d["labels"] == k).sum()) for k in range(5)] for n, d in (("train", train), ("validation", validation), ("test", test))},
                 "finite": True, "id_overlap": overlaps, "cached_test_logit_replay_max_abs_error": replay_error,
                 "train_role": "only split used to fit B/C", "validation_role": "integrity/provenance only; no fitting, selection, or early stopping", "test_role": "single predeclared archived confirmatory evaluation only; no fitting or selection"}
    (ROOT / "split_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    x, y = torch.tensor(train["features"], dtype=torch.float32, device=device), torch.tensor(train["labels"], dtype=torch.long, device=device)
    w, b = original_weight.to(device), original_bias.to(device)
    direction = DirectionOnlyLinear(w, b, w.norm(dim=1)).to(device)
    init_error = float((direction(x) - (x @ w.T + b)).abs().max().detach().cpu())
    if init_error > 2e-5: raise RuntimeError(f"C initialization replay error {init_error}")
    balanced = nn.Linear(512, 5).to(device)
    with torch.no_grad(): balanced.weight.copy_(w); balanced.bias.copy_(b)
    histories = {"B_balanced": fit(balanced, x, y, 1e-4), "C_direction_only": fit(direction, x, y, 0.0)}
    norm_error, bias_error = float(direction.max_norm_error().detach().cpu()), float((direction.fixed_bias - b).abs().max().detach().cpu())
    if max(norm_error, bias_error) > 1e-6: raise RuntimeError(f"C constraint failure norm={norm_error}, bias={bias_error}")
    evaluation = ROOT / "evaluation"; evaluation.mkdir()
    results = {}
    with torch.no_grad():
        for condition, head in (("A_original_ce", None), ("B_balanced", balanced), ("C_direction_only", direction)):
            logits = replay if head is None else head(torch.tensor(test["features"], dtype=torch.float32, device=device)).cpu().numpy()
            results[condition] = report(test["labels"], logits, test["sample_ids"], evaluation / condition)
    for condition, history in histories.items(): write_csv(ROOT / "histories" / f"{condition}.csv", history)
    states = ROOT / "head_states"; states.mkdir()
    torch.save({"condition": "A_original_ce", "weight": original_weight, "bias": original_bias, "frozen": True}, states / "A_original_ce.pt")
    torch.save({"condition": "B_balanced", "state_dict": balanced.cpu().state_dict(), "epochs": EPOCHS, "optimizer": "AdamW", "lr": .001, "weight_decay": 1e-4}, states / "B_balanced.pt")
    torch.save({"condition": "C_direction_only", "state_dict": direction.cpu().state_dict(), "epochs": EPOCHS, "optimizer": "AdamW", "lr": .001, "weight_decay": 0.0, "fixed_bias": original_bias, "fixed_norms": direction.fixed_norms.cpu()}, states / "C_direction_only.pt")
    weights = {"A": w.detach(), "B": balanced.weight.detach(), "C": direction.effective_weight().detach()}
    rows = []
    for k in range(5):
        rows.append({"class": k, "norm_a": float(weights["A"][k].norm()), "norm_b": float(weights["B"][k].norm()), "norm_c": float(weights["C"][k].norm()), "bias_a": float(b[k]), "bias_b": float(balanced.bias[k].detach()), "bias_c": float(direction.fixed_bias[k].detach()), "delta_norm_b_a": float(weights["B"][k].norm()-weights["A"][k].norm()), "delta_bias_b_a": float(balanced.bias[k].detach()-b[k]), "cos_b_a": cosine(weights["B"][k], weights["A"][k]), "cos_c_a": cosine(weights["C"][k], weights["A"][k])})
    write_csv(ROOT / "parameter_diagnostics.csv", rows)
    # X logits: exact adjacent upper-interior class is M / class 3.
    margin_rows = []
    xmask = test["labels"] == 4
    for condition in ("A_original_ce", "B_balanced", "C_direction_only"):
        with np.load(evaluation / condition / "predictions.npz", allow_pickle=False) as p: margin = p["logits"][xmask, 4] - p["logits"][xmask, 3]
        margin_rows.append({"condition": condition, "margin": "z_x_minus_z_m", "mean": float(margin.mean()), "median": float(np.median(margin)), "positive_fraction": float((margin > 0).mean())})
    write_csv(ROOT / "x_logit_margins.csv", margin_rows)
    write_csv(ROOT / "x_representation_subgroups.csv", x_geometry_subgroups(train, test, evaluation))
    protocol = {"conditions": ["A original frozen CE", "B full balanced CE head", "C fixed-A-norm/bias direction-only balanced CE"], "backbone_retrained": False, "head_fit_split": "train only", "confirmation_split": "archived test, one predeclared evaluation", "validation_selection": "none", "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "learning_rate": .001, "balanced_weight_decay": 1e-4, "direction_weight_decay": 0.0, "seed": 0, "c_initialization_replay_max_abs_error": init_error, "max_c_norm_error": norm_error, "max_c_bias_error": bias_error}
    (ROOT / "summary.json").write_text(json.dumps({"protocol": protocol, "results": results}, indent=2) + "\n")
    print(json.dumps({"output": str(ROOT), "c_initialization_replay_max_abs_error": init_error, "max_c_norm_error": norm_error, "max_c_bias_error": bias_error}, indent=2), flush=True)


if __name__ == "__main__": main()
