#!/usr/bin/env python3
"""Frozen A/C mechanism replication runner for RetinaMNIST and Solar.

A = original frozen classifier head.
C = direction-only balanced-CE head with original row norms and biases fixed.

RetinaMNIST uses training-only five-fold OOF head evaluation on one fixed
backbone representation.  Solar fits C on aligned training features and
reuses the fixed archived readout for backbone-realization robustness.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices, stratified_five_fold_assignments
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.models.resnet import make_resnet18

REPLAY_TOL = 2e-5
CONSTRAINT_TOL = 1e-6
HEAD_LR = 1e-3
HEAD_BATCH = 64
HEAD_EPOCHS = 100
FOLD_SEED = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved.get("model_state_dict", saved))
    if state is None or "fc.weight" not in state or "fc.bias" not in state:
        raise ValueError(f"checkpoint lacks canonical fc head: {checkpoint}")
    if tuple(state["fc.weight"].shape) != (5, 512) or tuple(state["fc.bias"].shape) != (5,):
        raise ValueError("expected 5x512 linear head")
    return state


def capture_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    batches: list[torch.Tensor] = []

    def hook(_module, inputs):
        batches.append(inputs[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(hook)
    logits, labels, ids = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, sample_id in loader:
            logits.append(model(x.to(device, non_blocking=True)).detach().cpu())
            labels.append(y.reshape(-1).detach().cpu())
            ids.append(sample_id.reshape(-1).detach().cpu())
    handle.remove()
    return {
        "features": torch.cat(batches).numpy().astype(np.float32),
        "logits": torch.cat(logits).numpy().astype(np.float32),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "sample_ids": torch.cat(ids).numpy().astype(np.int64),
    }


def make_retina_data(root: str, transform_name: str, batch: int, workers: int):
    """Load the official local native-28 training split without package drift.

    The provenance audit already established this NPZ as the official local
    RetinaMNIST artifact.  Loading it directly avoids relying on an unavailable
    ``medmnist`` installation while preserving ToTensor and Normalize behavior.
    """
    archive_path = Path(root) / "retinamnist.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"missing official RetinaMNIST NPZ: {archive_path}")
    with np.load(archive_path) as archive:
        images = archive["train_images"]
        labels = archive["train_labels"].reshape(-1).astype(np.int64)
    if images.shape != (1080, 28, 28, 3) or labels.shape != (1080,):
        raise ValueError(f"unexpected official Retina train split: {images.shape}, {labels.shape}")
    if transform_name not in ("normalize_half", "to_tensor"):
        raise ValueError(f"unknown Retina transform: {transform_name}")

    class OfficialRetinaTrain(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            image = torch.from_numpy(images[index]).permute(2, 0, 1).to(torch.float32).div(255.0)
            if transform_name == "normalize_half":
                image = image.sub(0.5).div(0.5)
            return image, torch.tensor(labels[index], dtype=torch.long), torch.tensor(index, dtype=torch.long)

    return DataLoader(OfficialRetinaTrain(), batch_size=batch, shuffle=False, num_workers=workers)


def make_solar_data(root: str, index: str, stats_path: str, batch: int, workers: int):
    from phase3_7a_solar_3ch import EXPECTED, Solar, manifest, source_channels
    from phase3_8_solar_confirmation import verify_stats

    source_channels(root)
    stats = verify_stats(Path(stats_path))
    names = ("train", "validation", "test")
    frames = [manifest(Path(index) / f"{name}.csv", expected) for name, expected in zip(names, EXPECTED)]
    datasets = [Solar(frame, root, (stats["mean"], stats["std"]), augment=False) for frame in frames]
    id_sets = [set(int(x) for x in ds.d.id.tolist()) for ds in datasets]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise RuntimeError("Solar split ID overlap")
    loaders = [DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
               for ds in datasets]
    return loaders[0], loaders[2]


def build_model(dataset: str, state: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    if dataset == "retina":
        model = make_resnet18(5)
    else:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 5)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def replay_check(features: np.ndarray, model_logits: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> float:
    # Keep the replay arithmetic in PyTorch float32, matching the model's
    # classifier path.  A NumPy BLAS matmul can introduce a larger accumulation
    # difference than the frozen replay tolerance on otherwise identical data.
    with torch.no_grad():
        head_logits = torch.nn.functional.linear(torch.from_numpy(features), weight.cpu(), bias.cpu()).numpy()
    err = float(np.abs(head_logits - model_logits).max())
    if err > REPLAY_TOL:
        raise RuntimeError(f"A cached/full-model replay failed: {err}")
    return err


def fit_direction_head(
    features: np.ndarray,
    labels: np.ndarray,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[DirectionOnlyLinear, list[dict[str, float | int]], float, float, float]:
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    head = DirectionOnlyLinear(weight.to(device), bias.to(device)).to(device)
    with torch.no_grad():
        original = x[: min(256, len(x))].to(device) @ weight.to(device).T + bias.to(device)
        init_err = float((head(x[: min(256, len(x))].to(device)) - original).abs().max().cpu())
    if init_err > REPLAY_TOL:
        raise RuntimeError(f"C initialization replay failed: {init_err}")

    optimizer = torch.optim.AdamW([head.direction], lr=HEAD_LR, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, HEAD_EPOCHS + 1):
        losses = []
        for _ in range(int(np.ceil(len(y) / HEAD_BATCH))):
            idx = balanced_batch_indices(y, HEAD_BATCH, generator)
            logits = head(x[idx].to(device))
            loss = torch.nn.functional.cross_entropy(logits, y[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "balanced_ce": float(np.mean(losses))})

    norm_err = float(head.max_norm_error().detach().cpu())
    bias_err = float((head.fixed_bias.detach().cpu() - bias.detach().cpu()).abs().max())
    if norm_err > CONSTRAINT_TOL or bias_err > CONSTRAINT_TOL:
        raise RuntimeError(f"C constraint failure norm={norm_err} bias={bias_err}")
    return head, history, init_err, norm_err, bias_err


def geometry(fit_x: np.ndarray, fit_y: np.ndarray, eval_x: np.ndarray) -> dict[str, np.ndarray]:
    centroids = np.stack([fit_x[fit_y == k].mean(axis=0) for k in range(5)])
    distances = np.sqrt(((eval_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(distances, axis=1)
    nearest = order[:, 0]
    d1 = distances[np.arange(len(distances)), order[:, 0]]
    d2 = distances[np.arange(len(distances)), order[:, 1]]
    return {
        "nearest_centroid": nearest.astype(np.int64),
        "endpoint_adj_margin": (distances[:, 3] - distances[:, 4]).astype(np.float64),
        "generic_centroid_margin": (d2 - d1).astype(np.float64),
    }


def make_outputs(labels: np.ndarray, a_logits: np.ndarray, c_logits: np.ndarray) -> dict[str, np.ndarray]:
    a_prob, c_prob = softmax(a_logits), softmax(c_logits)
    a_dec, c_dec = bayes_decisions(a_prob), bayes_decisions(c_prob)
    classes = np.arange(5)
    a_mean = a_prob @ classes
    c_mean = c_prob @ classes
    return {
        "a_logits": a_logits,
        "c_logits": c_logits,
        "a_probabilities": a_prob,
        "c_probabilities": c_prob,
        "a_l1": a_dec["l1_bayes_decision"].astype(np.int64),
        "c_l1": c_dec["l1_bayes_decision"].astype(np.int64),
        "a_predictive_mean": a_mean,
        "c_predictive_mean": c_mean,
        "a_inward_shrinkage": 4.0 - a_mean,
        "c_inward_shrinkage": 4.0 - c_mean,
    }


def endpoint_summary(labels: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, object]:
    mask = labels == 4
    if not mask.any():
        raise RuntimeError("no rare-end class-4 samples")
    a_err = np.abs(4 - values["a_l1"][mask])
    c_err = np.abs(4 - values["c_l1"][mask])
    return {
        "support": int(mask.sum()),
        "a_mae": float(a_err.mean()),
        "c_mae": float(c_err.mean()),
        "delta_mae_c_minus_a": float(c_err.mean() - a_err.mean()),
        "a_exact": int((values["a_l1"][mask] == 4).sum()),
        "c_exact": int((values["c_l1"][mask] == 4).sum()),
        "a_mean_shrinkage": float(values["a_inward_shrinkage"][mask].mean()),
        "c_mean_shrinkage": float(values["c_inward_shrinkage"][mask].mean()),
    }


def save_per_sample(out: Path, ids, labels, folds, geom, values) -> None:
    np.savez_compressed(
        out / "per_sample_arrays.npz",
        sample_ids=ids,
        labels=labels,
        folds=folds,
        nearest_centroid=geom["nearest_centroid"],
        endpoint_adj_margin=geom["endpoint_adj_margin"],
        generic_centroid_margin=geom["generic_centroid_margin"],
        **values,
    )
    fields = [
        "sample_id", "label", "fold", "nearest_centroid",
        "endpoint_adj_margin", "generic_centroid_margin",
        "a_l1", "c_l1", "a_predictive_mean", "c_predictive_mean",
        "a_inward_shrinkage", "c_inward_shrinkage",
    ]
    with (out / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(labels)):
            writer.writerow({
                "sample_id": int(ids[i]),
                "label": int(labels[i]),
                "fold": int(folds[i]),
                "nearest_centroid": int(geom["nearest_centroid"][i]),
                "endpoint_adj_margin": float(geom["endpoint_adj_margin"][i]),
                "generic_centroid_margin": float(geom["generic_centroid_margin"][i]),
                "a_l1": int(values["a_l1"][i]),
                "c_l1": int(values["c_l1"][i]),
                "a_predictive_mean": float(values["a_predictive_mean"][i]),
                "c_predictive_mean": float(values["c_predictive_mean"][i]),
                "a_inward_shrinkage": float(values["a_inward_shrinkage"][i]),
                "c_inward_shrinkage": float(values["c_inward_shrinkage"][i]),
            })


def run_retina(args, model, state, device, out):
    loader = make_retina_data(args.retina_root, args.retina_transform, args.extract_batch, args.workers)
    data = capture_features(model, loader, device)
    if data["features"].shape != (1080, 512):
        raise RuntimeError(f"unexpected Retina training feature shape: {data['features'].shape}")
    if np.bincount(data["labels"], minlength=5).tolist() != [486, 128, 206, 194, 66]:
        raise RuntimeError("Retina canonical train counts mismatch")

    weight, bias = state["fc.weight"].detach().float(), state["fc.bias"].detach().float()
    replay = replay_check(data["features"], data["logits"], weight, bias)
    folds = stratified_five_fold_assignments(data["labels"], seed=FOLD_SEED)
    a_logits = data["logits"].copy()
    c_logits = np.empty_like(a_logits)
    nearest = np.empty(len(folds), dtype=np.int64)
    endpoint_margin = np.empty(len(folds), dtype=np.float64)
    generic_margin = np.empty(len(folds), dtype=np.float64)
    max_init = max_norm = max_bias = 0.0

    states = out / "head_states"
    states.mkdir()
    histories = out / "histories"
    histories.mkdir()

    for fold in range(5):
        fit = folds != fold
        held = folds == fold
        head, history, init_err, norm_err, bias_err = fit_direction_head(
            data["features"][fit], data["labels"][fit], weight, bias,
            seed=10_000 + fold, device=device,
        )
        with torch.no_grad():
            c_logits[held] = head(torch.as_tensor(data["features"][held], dtype=torch.float32, device=device)).cpu().numpy()
        g = geometry(data["features"][fit], data["labels"][fit], data["features"][held])
        nearest[held] = g["nearest_centroid"]
        endpoint_margin[held] = g["endpoint_adj_margin"]
        generic_margin[held] = g["generic_centroid_margin"]
        max_init, max_norm, max_bias = max(max_init, init_err), max(max_norm, norm_err), max(max_bias, bias_err)
        torch.save({
            "condition": "C_direction_only",
            "fold": fold,
            "state_dict": head.cpu().state_dict(),
            "fixed_norms": head.fixed_norms.cpu(),
            "fixed_bias": head.fixed_bias.cpu(),
        }, states / f"C_fold_{fold}.pt")
        with (histories / f"fold_{fold}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)

    geom = {"nearest_centroid": nearest, "endpoint_adj_margin": endpoint_margin, "generic_centroid_margin": generic_margin}
    values = make_outputs(data["labels"], a_logits, c_logits)
    save_per_sample(out, data["sample_ids"], data["labels"], folds, geom, values)
    return endpoint_summary(data["labels"], values), {
        "a_replay_max_abs_error": replay,
        "c_initialization_replay_max_abs_error": max_init,
        "c_max_norm_error": max_norm,
        "c_max_bias_error": max_bias,
        "oof": "training-only five-fold head evaluation on a fixed learned representation",
        "centroid_rule": "per-fold centroids from head-fitting features only; held-fold geometry evaluated out-of-fold",
    }


def run_solar(args, model, state, device, out):
    train_loader, test_loader = make_solar_data(args.solar_root, args.solar_index, args.solar_stats, args.extract_batch, args.workers)
    train = capture_features(model, train_loader, device)
    test = capture_features(model, test_loader, device)
    if train["features"].shape[1] != 512 or test["features"].shape[1] != 512:
        raise RuntimeError("unexpected Solar feature dimension")
    if set(train["sample_ids"].tolist()) & set(test["sample_ids"].tolist()):
        raise RuntimeError("Solar train/test ID overlap")

    weight, bias = state["fc.weight"].detach().float(), state["fc.bias"].detach().float()
    replay = replay_check(test["features"], test["logits"], weight, bias)
    a_logits = test["features"] @ weight.numpy().T + bias.numpy()
    head, history, init_err, norm_err, bias_err = fit_direction_head(
        train["features"], train["labels"], weight, bias, seed=10_000, device=device,
    )
    with torch.no_grad():
        c_logits = head(torch.as_tensor(test["features"], dtype=torch.float32, device=device)).cpu().numpy()
    geom = geometry(train["features"], train["labels"], test["features"])
    values = make_outputs(test["labels"], a_logits, c_logits)
    folds = np.full(len(test["labels"]), -1, dtype=np.int64)
    save_per_sample(out, test["sample_ids"], test["labels"], folds, geom, values)
    torch.save({
        "condition": "C_direction_only",
        "state_dict": head.cpu().state_dict(),
        "fixed_norms": head.fixed_norms.cpu(),
        "fixed_bias": head.fixed_bias.cpu(),
    }, out / "C_head.pt")
    with (out / "head_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    return endpoint_summary(test["labels"], values), {
        "a_replay_max_abs_error": replay,
        "c_initialization_replay_max_abs_error": init_err,
        "c_max_norm_error": norm_err,
        "c_max_bias_error": bias_err,
        "evaluation_population": "reused archived aligned Solar readout; not a new external confirmation",
        "centroid_rule": "aligned training features only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("retina", "solar"), required=True)
    parser.add_argument("--objective", choices=("ce", "rps"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extract-batch", type=int, default=128)
    parser.add_argument("--retina-root", default="data/medmnist")
    parser.add_argument("--retina-transform", choices=("normalize_half", "to_tensor"), default="normalize_half")
    parser.add_argument("--solar-root", default="/scratch/users/jhong36/data/surya-bench-224.zarr")
    parser.add_argument("--solar-index", default="/scratch/users/jhong36/data")
    parser.add_argument("--solar-stats")
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    if args.dataset == "solar" and not args.solar_stats:
        parser.error("--solar-stats is required for Solar")

    seed_everything(args.seed)
    device = torch.device(args.device)
    state = load_state(args.checkpoint)
    model = build_model(args.dataset, state, device)

    # Persist A explicitly so every condition has an auditable original-head state.
    torch.save({
        "condition": "A_original",
        "weight": state["fc.weight"].detach().cpu(),
        "bias": state["fc.bias"].detach().cpu(),
        "frozen": True,
    }, args.out / "A_original_head.pt")

    if args.dataset == "retina":
        summary, integrity = run_retina(args, model, state, device, args.out)
    else:
        summary, integrity = run_solar(args, model, state, device, args.out)

    manifest = {
        "protocol": "FROZEN FOR EXECUTION A/C mechanism replication",
        "dataset": args.dataset,
        "backbone_objective": args.objective,
        "backbone_seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "feature_definition": "512-D input to model.fc after average pooling/flatten",
        "retina_data_provenance": (
            {"source": str(Path(args.retina_root) / "retinamnist.npz"), "split": "official train"}
            if args.dataset == "retina"
            else None
        ),
        "retina_preprocessing": args.retina_transform if args.dataset == "retina" else None,
        "seed_role": "historical/hypothesis-forming" if args.seed == 0 else "confirmatory",
        "A": "original frozen head",
        "C": {
            "objective": "cross_entropy",
            "sampling": "replacement class-balanced",
            "optimizer": "AdamW",
            "learning_rate": HEAD_LR,
            "batch_size": HEAD_BATCH,
            "epochs": HEAD_EPOCHS,
            "direction_weight_decay": 0.0,
            "fixed_original_norms": True,
            "fixed_original_biases": True,
        },
        "integrity_tolerances": {
            "a_replay_max_abs_error": REPLAY_TOL,
            "c_initialization_replay_max_abs_error": REPLAY_TOL,
            "c_norm_error": CONSTRAINT_TOL,
            "c_bias_error": CONSTRAINT_TOL,
        },
        "integrity": integrity,
        "rare_end_h1_summary": summary,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
