#!/usr/bin/env python3
"""Fit Solar natural-sampling direction-only N heads from frozen features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import natural_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions


FEATURE_ROOT = Path("outputs/mechanism_replication/features/solar")
AC_ROOT = Path("outputs/mechanism_replication/ac/solar")
OUT_ROOT = Path("outputs/mechanism_replication/natural_direction/solar")

LR = 1e-3
BATCH = 64
EPOCHS = 100
TOL = 1e-6
K = 5
ENDPOINT = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def mode_and_l1(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    l1 = bayes_decisions(probabilities)["l1_bayes_decision"].astype(np.int64)
    mode = probabilities.argmax(axis=1).astype(np.int64)
    return mode, l1


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_feature_archive(
    path: Path,
    expected_sha256: str,
) -> dict[str, np.ndarray]:
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"feature archive SHA256 mismatch: {path}")

    with np.load(path, allow_pickle=False) as data:
        result = {
            "features": data["features"],
            "labels": data["labels"],
            "sample_ids": data["sample_ids"],
        }

    features = result["features"]
    labels = result["labels"]
    ids = result["sample_ids"]

    if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 512:
        raise RuntimeError(f"invalid frozen feature matrix: {path}: {features.shape}")
    if labels.dtype.kind not in ("i", "u") or labels.shape != (len(features),):
        raise RuntimeError(f"invalid labels: {path}")
    if ids.shape != (len(features),) or len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"invalid or duplicate sample IDs: {path}")

    return result


def assert_archived_reference_identity(
    labels: np.ndarray,
    archived: dict[str, np.ndarray],
) -> None:
    for condition in ("a", "c"):
        probabilities = archived[f"{condition}_probabilities"]
        _, computed_l1 = mode_and_l1(probabilities)

        if not np.array_equal(computed_l1, archived[f"{condition}_l1"]):
            raise RuntimeError(
                f"archived {condition.upper()} exact-L1 decisions are inconsistent"
            )

        predictive_mean = probabilities @ np.arange(K)
        if not np.allclose(
            predictive_mean,
            archived[f"{condition}_predictive_mean"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"archived {condition.upper()} predictive means are inconsistent"
            )

        if not np.array_equal(labels, archived["labels"]):
            raise RuntimeError(f"archived {condition.upper()} labels are inconsistent")


def fit_natural(
    features: np.ndarray,
    labels: np.ndarray,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
) -> tuple[
    DirectionOnlyLinear,
    list[dict[str, object]],
    np.ndarray,
    dict[str, float],
]:
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)

    head = DirectionOnlyLinear(weight.to(device), bias.to(device)).to(device)

    if head.fixed_norms.requires_grad or head.fixed_bias.requires_grad:
        raise RuntimeError("fixed norms or fixed biases unexpectedly require gradients")

    optimizer = torch.optim.AdamW(
        [head.direction],
        lr=LR,
        weight_decay=0.0,
    )

    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["params"] != [
        head.direction
    ]:
        raise RuntimeError("optimizer contains parameters other than directions")

    before_bias = head.fixed_bias.detach().cpu().clone()
    before_norms = head.fixed_norms.detach().cpu().clone()

    # Matches the archived Solar C runner's random-seed convention. The only
    # changed operation is natural_batch_indices rather than balanced_batch_indices.
    generator = torch.Generator().manual_seed(10_000)

    observed_counts = np.zeros(K, dtype=np.int64)
    history: list[dict[str, object]] = []

    for epoch in range(1, EPOCHS + 1):
        losses: list[float] = []
        steps = 0

        # Each epoch is precisely one shuffled permutation: no replacement,
        # class weighting, oversampling, undersampling, or balanced sampler.
        for index in natural_batch_indices(len(y), BATCH, generator):
            observed_counts += np.bincount(
                y[index].numpy(),
                minlength=K,
            )

            logits = head(x[index].to(device))
            loss = torch.nn.functional.cross_entropy(
                logits,
                y[index].to(device),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            steps += 1

        history.append(
            {
                "epoch": epoch,
                "natural_ce": float(np.mean(losses)),
                "steps": steps,
            }
        )

    expected_counts = np.bincount(labels, minlength=K) * EPOCHS
    if not np.array_equal(observed_counts, expected_counts):
        raise RuntimeError(
            "natural sampler did not preserve empirical frequencies: "
            f"observed={observed_counts.tolist()} "
            f"expected={expected_counts.tolist()}"
        )

    constraints = {
        "max_norm_error": float(head.max_norm_error().detach().cpu()),
        "max_bias_error": float(
            (head.fixed_bias.detach().cpu() - before_bias).abs().max()
        ),
        "stored_norm_error": float(
            (head.fixed_norms.detach().cpu() - before_norms).abs().max()
        ),
    }

    if max(constraints.values()) > TOL:
        raise RuntimeError(f"fixed-parameter assertion failed: {constraints}")

    return head, history, observed_counts, constraints


def run(args: argparse.Namespace) -> dict[str, object]:
    feature_dir = FEATURE_ROOT / args.objective / f"seed_{args.seed}"
    ac_dir = AC_ROOT / args.objective / f"seed_{args.seed}"
    output = OUT_ROOT / args.objective / f"seed_{args.seed}"

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    feature_manifest_path = feature_dir / "feature_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())

    if feature_manifest["a_output_reproduction"]["status"] != "PASS":
        raise RuntimeError("Phase-A A-output reproduction is not PASS")

    if feature_manifest.get("checkpoint_integrity") != "READY":
        raise RuntimeError("checkpoint integrity is not READY")

    train = load_feature_archive(
        feature_dir / "train_features.npz",
        feature_manifest["splits"]["train"]["sha256"],
    )
    validation = load_feature_archive(
        feature_dir / "val_features.npz",
        feature_manifest["splits"]["val"]["sha256"],
    )
    evaluation = load_feature_archive(
        feature_dir / "eval_features.npz",
        feature_manifest["splits"]["eval"]["sha256"],
    )

    for split_name, values in (
        ("train", train),
        ("val", validation),
        ("eval", evaluation),
    ):
        recorded = feature_manifest["splits"][split_name]
        if len(values["labels"]) != recorded["rows"]:
            raise RuntimeError(
                f"{split_name} row count differs from Phase-A manifest"
            )
        observed_class_counts = np.bincount(values["labels"], minlength=K).tolist()
        if observed_class_counts != recorded["class_counts"]:
            raise RuntimeError(
                f"{split_name} class counts differ from Phase-A manifest"
            )

    train_ids = set(train["sample_ids"].tolist())
    validation_ids = set(validation["sample_ids"].tolist())
    eval_ids = set(evaluation["sample_ids"].tolist())

    if train_ids & validation_ids or train_ids & eval_ids or validation_ids & eval_ids:
        raise RuntimeError("train/validation/eval feature populations overlap")

    with np.load(ac_dir / "per_sample_arrays.npz", allow_pickle=False) as source:
        archived = {
            key: source[key]
            for key in (
                "sample_ids",
                "labels",
                "a_logits",
                "c_logits",
                "a_probabilities",
                "c_probabilities",
                "a_l1",
                "c_l1",
                "a_predictive_mean",
                "c_predictive_mean",
            )
        }

    if not np.array_equal(evaluation["sample_ids"], archived["sample_ids"]):
        raise RuntimeError("eval feature and archived A/C sample IDs differ")

    if not np.array_equal(evaluation["labels"], archived["labels"]):
        raise RuntimeError("eval feature and archived A/C labels differ")

    assert_archived_reference_identity(evaluation["labels"], archived)

    ac_manifest = json.loads((ac_dir / "manifest.json").read_text())
    c_protocol = ac_manifest.get("C", {})
    expected_c = {
        "objective": "cross_entropy",
        "sampling": "replacement class-balanced",
        "optimizer": "AdamW",
        "learning_rate": LR,
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "direction_weight_decay": 0.0,
        "fixed_original_norms": True,
        "fixed_original_biases": True,
    }
    if any(c_protocol.get(key) != value for key, value in expected_c.items()):
        raise RuntimeError(
            "archived C protocol cannot be reconstructed exactly; refusing N fit"
        )

    head_a = torch.load(
        ac_dir / "A_original_head.pt",
        map_location="cpu",
        weights_only=False,
    )
    weight = head_a["weight"].float()
    bias = head_a["bias"].float()

    if tuple(weight.shape) != (K, 512) or tuple(bias.shape) != (K,):
        raise RuntimeError("invalid original Solar A head")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    head, history, observed_counts, constraints = fit_natural(
        train["features"],
        train["labels"],
        weight,
        bias,
        device,
    )

    with torch.inference_mode():
        n_logits = (
            head(
                torch.as_tensor(
                    evaluation["features"],
                    dtype=torch.float32,
                    device=device,
                )
            )
            .cpu()
            .numpy()
        )

    n_probabilities = softmax(n_logits)
    n_mode, n_l1 = mode_and_l1(n_probabilities)
    a_mode, a_l1 = mode_and_l1(archived["a_probabilities"])
    c_mode, c_l1 = mode_and_l1(archived["c_probabilities"])

    classes = np.arange(K)
    n_predictive_mean = n_probabilities @ classes
    endpoint = evaluation["labels"] == ENDPOINT

    metrics = {
        "endpoint_mae_A": float(
            np.abs(evaluation["labels"][endpoint] - a_l1[endpoint]).mean()
        ),
        "endpoint_mae_N": float(
            np.abs(evaluation["labels"][endpoint] - n_l1[endpoint]).mean()
        ),
        "endpoint_mae_C": float(
            np.abs(evaluation["labels"][endpoint] - c_l1[endpoint]).mean()
        ),
    }
    metrics.update(
        {
            "delta_NA_endpoint": (
                metrics["endpoint_mae_N"] - metrics["endpoint_mae_A"]
            ),
            "delta_CA_endpoint": (
                metrics["endpoint_mae_C"] - metrics["endpoint_mae_A"]
            ),
            "delta_NC_endpoint": (
                metrics["endpoint_mae_N"] - metrics["endpoint_mae_C"]
            ),
        }
    )

    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    torch.save(
        {
            "condition": "N_natural_direction_only",
            "state_dict": head.cpu().state_dict(),
            "fixed_norms": head.fixed_norms.cpu(),
            "fixed_bias": head.fixed_bias.cpu(),
        },
        output / "N_head.pt",
    )

    save_csv(output / "training_history.csv", history)

    np.savez_compressed(
        output / "per_sample_arrays.npz",
        sample_ids=evaluation["sample_ids"],
        labels=evaluation["labels"],
        a_logits=archived["a_logits"],
        n_logits=n_logits,
        c_logits=archived["c_logits"],
        a_probabilities=archived["a_probabilities"],
        n_probabilities=n_probabilities,
        c_probabilities=archived["c_probabilities"],
        a_mode=a_mode,
        n_mode=n_mode,
        c_mode=c_mode,
        a_l1=a_l1,
        n_l1=n_l1,
        c_l1=c_l1,
        a_predictive_mean=archived["a_predictive_mean"],
        n_predictive_mean=n_predictive_mean,
        c_predictive_mean=archived["c_predictive_mean"],
    )

    config = {
        "condition": "N natural empirical-sampling direction-only",
        "adaptation_objective": "cross_entropy",
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": 0.0,
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "initialization": "original frozen A classifier head",
        "fixed_original_norms": True,
        "fixed_original_biases": True,
        "sampling": (
            "natural shuffled empirical frozen-training distribution; "
            "no class weighting, replacement sampling, oversampling, "
            "undersampling, logit adjustment, or bias adjustment"
        ),
        "checkpoint_selection": (
            "none; fixed terminal epoch 100 exactly matches archived Solar C"
        ),
        "validation_role": (
            "validation features are exported and count-audited only; "
            "they are not used for selection because archived C has none"
        ),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    manifest = {
        "dataset": "solar",
        "objective": args.objective,
        "seed": args.seed,
        "condition": "N natural empirical-sampling direction-only",
        "feature_manifest": str(feature_manifest_path),
        "feature_manifest_sha256": sha256(feature_manifest_path),
        "source_ac": str(ac_dir),
        "source_ac_manifest": str(ac_dir / "manifest.json"),
        "source_c_protocol": c_protocol,
        "a_c_identity": "PASS",
        "checkpoint_path": feature_manifest["checkpoint_path"],
        "checkpoint_sha256": feature_manifest["checkpoint_sha256"],
        "selected_backbone_epoch": feature_manifest["selected_epoch"],
        "train_rows": int(len(train["labels"])),
        "validation_rows": int(len(validation["labels"])),
        "eval_rows": int(len(evaluation["labels"])),
        "train_class_counts": np.bincount(
            train["labels"],
            minlength=K,
        ).tolist(),
        "observed_draw_counts_across_100_epochs": observed_counts.tolist(),
        "constraints": constraints,
        "selected_head_epoch": EPOCHS,
        "validation_loss": None,
        "head_steps": int(sum(int(row["steps"]) for row in history)),
        "cluster_job_id": os.getenv("SLURM_JOB_ID", "unset"),
        "no_backbone_training": True,
        "no_checkpoint_change": True,
        "no_a_refit": True,
        "no_c_refit": True,
        "only_directions_updated": True,
        "metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("ce", "rps"), required=True)
    parser.add_argument("--seed", choices=(1, 2, 3, 4), type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
