#!/usr/bin/env python3
"""Frozen Phase 3.15 Solar RPS A/B/C/D direction-scale confirmation."""
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


ROOT = Path("outputs/solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24")
FEATURE_ROOT = Path("outputs/solar/phase3_9_mechanism_audit/features/rps")
CHECKPOINT = Path("outputs/solar/phase3_8_shrinkage_confirmation/rps/seed_0/selected_checkpoint.pt")
ALPHA = 0.50
BATCH_SIZE = 64
EPOCHS = 100


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cache(split: str) -> dict[str, np.ndarray]:
    path = FEATURE_ROOT / f"{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "sample_ids", "logits"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}")
        data = {key: archive[key].copy() for key in archive.files}
    if data["features"].ndim != 2 or data["features"].shape[1] != 512:
        raise ValueError(f"unexpected frozen feature shape: {data['features'].shape}")
    if not np.isfinite(data["features"]).all() or not np.isfinite(data["logits"]).all():
        raise ValueError(f"non-finite cached values in {path}")
    if not np.isin(data["labels"], np.arange(5)).all():
        raise ValueError(f"invalid labels in {path}")
    if len(np.unique(data["sample_ids"])) != len(data["sample_ids"]):
        raise ValueError(f"non-unique sample IDs in {path}")
    return data


def checkpoint_head() -> tuple[torch.Tensor, torch.Tensor]:
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved.get("model_state_dict", saved))
    weight, bias = state["fc.weight"].float(), state["fc.bias"].float()
    if weight.shape != (5, 512) or bias.shape != (5,):
        raise ValueError(f"unexpected RPS head shapes {weight.shape}/{bias.shape}")
    return weight, bias


def fit(head: nn.Module, x: torch.Tensor, y: torch.Tensor, weight_decay: float) -> list[dict]:
    params = head.parameters() if isinstance(head, nn.Linear) else [head.direction]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(10000)
    history = []
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for _ in range(int(np.ceil(len(y) / BATCH_SIZE))):
            index = balanced_batch_indices(y, BATCH_SIZE, generator).to(x.device)
            loss = torch.nn.functional.cross_entropy(head(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "balanced_ce": float(np.mean(losses))})
    return history


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.detach().cpu()[None], b.detach().cpu()[None]).item())


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Phase 3.15; refusing CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = torch.device("cuda:0")
    print(json.dumps({"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_available": True, "gpu": torch.cuda.get_device_name(device), "python": sys.executable, "torch": torch.__version__}, indent=2), flush=True)

    train, validation, test = (load_cache(name) for name in ("train", "validation", "test"))
    ids = {name: set(data["sample_ids"].tolist()) for name, data in (("train", train), ("validation", validation), ("test", test))}
    overlaps = {"train_validation": len(ids["train"] & ids["validation"]), "train_test": len(ids["train"] & ids["test"]), "validation_test": len(ids["validation"] & ids["test"])}
    if any(overlaps.values()):
        raise ValueError(f"split ID overlap: {overlaps}")
    original_weight, original_bias = checkpoint_head()
    replay = test["features"] @ original_weight.numpy().T + original_bias.numpy()
    replay_error = float(np.abs(replay - test["logits"]).max())
    if replay_error > 2e-5:
        raise ValueError(f"cached-test logit replay error {replay_error}")
    ROOT.mkdir(parents=True)
    integrity = {"protocol": "Phase 3.15 frozen Solar RPS A/B/C/D", "feature_source": str(FEATURE_ROOT), "checkpoint": str(CHECKPOINT), "feature_dimension": 512, "counts": {name: int(len(data["labels"])) for name, data in (("train", train), ("validation", validation), ("test", test))}, "class_counts": {name: [int((data["labels"] == k).sum()) for k in range(5)] for name, data in (("train", train), ("validation", validation), ("test", test))}, "finite": True, "id_overlap": overlaps, "test_loaded": True, "test_role": "single predeclared archived confirmatory evaluation only; never used for fitting or selection", "validation_loaded": True, "validation_role": "integrity/provenance only; not used for fitting, selection, or early stopping", "train_role": "only split used to fit B/C/D", "cached_test_logit_replay_max_abs_error": replay_error, "device": str(device)}
    (ROOT / "split_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

    x = torch.tensor(train["features"], dtype=torch.float32, device=device)
    y = torch.tensor(train["labels"], dtype=torch.long, device=device)
    w, b = original_weight.to(device), original_bias.to(device)
    balanced = nn.Linear(512, 5).to(device)
    with torch.no_grad():
        balanced.weight.copy_(w); balanced.bias.copy_(b)
    histories = {"B_balanced": fit(balanced, x, y, 1e-4)}
    balanced_weight = balanced.weight.detach().clone()
    target_norms = (1 - ALPHA) * w.norm(dim=1) + ALPHA * balanced_weight.norm(dim=1)
    direction = DirectionOnlyLinear(w, b, w.norm(dim=1)).to(device)
    controlled = DirectionOnlyLinear(w, b, target_norms).to(device)
    histories["C_direction_only"] = fit(direction, x, y, 0.0)
    histories["D_controlled_scale_alpha_0.50"] = fit(controlled, x, y, 0.0)
    c_error, d_error = float(direction.max_norm_error().detach().cpu()), float(controlled.max_norm_error().detach().cpu())
    if max(c_error, d_error) > 1e-6:
        raise RuntimeError(f"fixed-norm integrity failure C={c_error} D={d_error}")

    heads: dict[str, nn.Module | None] = {"A_original_rps": None, "B_balanced": balanced, "C_direction_only": direction, "D_controlled_scale_alpha_0.50": controlled}
    evaluation = ROOT / "evaluation"
    evaluation.mkdir()
    results = {}
    with torch.no_grad():
        for condition, head in heads.items():
            logits = replay if head is None else head(torch.tensor(test["features"], dtype=torch.float32, device=device)).cpu().numpy()
            results[condition] = report(test["labels"], logits, test["sample_ids"], evaluation / condition)
    for condition, history in histories.items():
        write_csv(ROOT / "histories" / f"{condition}.csv", history)
    states = ROOT / "head_states"; states.mkdir()
    torch.save({"condition": "A_original_rps", "weight": original_weight, "bias": original_bias, "frozen": True}, states / "A_original_rps.pt")
    torch.save({"condition": "B_balanced", "state_dict": balanced.cpu().state_dict(), "epochs": EPOCHS, "optimizer": "AdamW", "lr": .001, "weight_decay": 1e-4}, states / "B_balanced.pt")
    for name, head in (("C_direction_only", direction), ("D_controlled_scale_alpha_0.50", controlled)):
        torch.save({"condition": name, "state_dict": head.cpu().state_dict(), "epochs": EPOCHS, "optimizer": "AdamW", "lr": .001, "weight_decay": 0.0, "fixed_bias": original_bias, "fixed_norms": head.fixed_norms.cpu()}, states / f"{name}.pt")
    weights = {"A": w.detach(), "B": balanced_weight, "C": direction.effective_weight().detach(), "D": controlled.effective_weight().detach()}
    rows = []
    for k in range(5):
        row = {"class": k, "norm_a": float(weights["A"][k].norm()), "norm_b": float(weights["B"][k].norm()), "norm_c": float(weights["C"][k].norm()), "norm_d": float(weights["D"][k].norm()), "bias_a": float(b[k]), "bias_b": float(balanced.bias[k].detach()), "delta_bias_b_a": float(balanced.bias[k].detach() - b[k])}
        for left, right in (("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "B")):
            row[f"cos_{left.lower()}_{right.lower()}"] = cosine(weights[left][k], weights[right][k])
        rows.append(row)
    write_csv(ROOT / "parameter_diagnostics.csv", rows)
    protocol = {"conditions": ["A original frozen RPS", "B full balanced CE head", "C fixed-A-norm/bias direction-only balanced CE", "D fixed alpha=.50 A/B-norm direction-only balanced CE"], "alpha": ALPHA, "rop": False, "backbone_retrained": False, "head_fit_split": "train only", "confirmation_split": "archived test, one predeclared evaluation", "validation_selection": "none", "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "learning_rate": .001, "balanced_weight_decay": 1e-4, "direction_weight_decay": 0.0, "seed": 0, "max_c_norm_error": c_error, "max_d_norm_error": d_error}
    (ROOT / "summary.json").write_text(json.dumps({"protocol": protocol, "results": results}, indent=2) + "\n")
    print(json.dumps({"output": str(ROOT), "max_c_norm_error": c_error, "max_d_norm_error": d_error}, indent=2), flush=True)


if __name__ == "__main__":
    main()
