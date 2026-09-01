#!/usr/bin/env python3
"""Frozen seed-0 CE/RPS penultimate-feature audit; performs inference only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision import transforms

from ordinal_uncertainty.evaluation.representation import (
    class_centroids,
    cosine_distances,
    euclidean_distances,
    l2_normalize,
    nearest_centroid,
    within_class_dispersion,
)
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.models.resnet import make_resnet18


ROOT = Path("outputs/retinamnist/native28")
METHODS = {
    "ce": {
        # The canonical native seed-0 run was reused by the resolution check.
        "checkpoint": Path("outputs/retinamnist/resolution_sanity_check/seed_0/size_28/best_checkpoint.pt"),
        "predictions": Path("outputs/retinamnist/resolution_sanity_check/seed_0/size_28/predictions.csv"),
        "transform_candidates": ("normalize_half",),
    },
    "rps": {
        "checkpoint": ROOT / "phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt",
        "predictions": ROOT / "phase2_model_comparison/rps/seed_0_artifact_complete/evaluation/predictions.csv",
        # The minimal config did not persist preprocessing; select only a
        # transform that exactly replays the selected checkpoint's saved logits.
        "transform_candidates": ("to_tensor", "normalize_half"),
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prediction_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ids = np.asarray([int(row["sample_id"]) for row in rows])
    labels = np.asarray([int(row["true_label"]) for row in rows])
    logits = np.asarray([json.loads(row["logits"]) for row in rows], dtype=np.float32)
    return ids, labels, logits


def make_transform(name: str) -> transforms.Compose:
    if name == "normalize_half":
        return transforms.Compose(
            [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
    if name == "to_tensor":
        return transforms.Compose([transforms.ToTensor()])
    raise ValueError(f"unknown transform: {name}")


def split_loaders(data_root: Path, batch_size: int, transform: transforms.Compose) -> dict[str, DataLoader]:
    from medmnist import RetinaMNIST

    datasets = {
        "train": RetinaMNIST(split="train", root=str(data_root), download=False, transform=transform),
        "val": RetinaMNIST(split="val", root=str(data_root), download=False, transform=transform),
        "test": RetinaMNIST(split="test", root=str(data_root), download=False, transform=transform),
    }
    return {split: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) for split, dataset in datasets.items()}


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved.get("state_dict"))
    if state is None:
        raise ValueError(f"checkpoint lacks a model state dictionary: {checkpoint_path}")
    model = make_resnet18(5)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def extract_split(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Capture the input to the final linear head during frozen forward inference."""
    feature_batches: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        feature_batches.append(inputs[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(capture)
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits.append(model(images.to(device)).detach().cpu())
            labels.append(batch_labels.reshape(-1).detach().cpu())
    handle.remove()
    values = torch.cat(logits).numpy().astype(np.float32)
    features = torch.cat(feature_batches).numpy().astype(np.float32)
    targets = torch.cat(labels).numpy().astype(np.int64)
    probabilities = torch.softmax(torch.from_numpy(values), dim=1).numpy().astype(np.float64)
    decisions = bayes_decisions(probabilities)
    return {
        "sample_id": np.arange(targets.size, dtype=np.int64),
        "labels": targets,
        "logits": values,
        "probabilities": probabilities,
        "features": features,
        **decisions,
    }


def decision_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    error = np.abs(labels - predictions)
    return {
        "accuracy": float((labels == predictions).mean()),
        "mae": float(error.mean()),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "severe_prevalence": float((error >= 2).mean()),
    }


def summarize_distances(
    method: str,
    space: str,
    split: str,
    labels: np.ndarray,
    distances: np.ndarray,
    true_class: int,
) -> list[dict[str, object]]:
    mask = labels == true_class
    rows = []
    for centroid_class in range(distances.shape[1]):
        values = distances[mask, centroid_class]
        rows.append(
            {
                "method": method,
                "space": space,
                "split": split,
                "true_class": true_class,
                "centroid_class": centroid_class,
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)),
            }
        )
    return rows


def margin_rows(
    method: str,
    space: str,
    split: str,
    labels: np.ndarray,
    distances: np.ndarray,
    true_class: int,
    reference_classes: tuple[int, ...],
) -> list[dict[str, object]]:
    own = distances[labels == true_class, true_class]
    rows = []
    for reference in reference_classes:
        margin = distances[labels == true_class, reference] - own
        rows.append(
            {
                "method": method,
                "space": space,
                "split": split,
                "true_class": true_class,
                "reference_class": reference,
                "count": int(margin.size),
                "mean": float(margin.mean()),
                "median": float(np.median(margin)),
                "fraction_negative": float((margin < 0).mean()),
                "fraction_positive": float((margin > 0).mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/medmnist"))
    parser.add_argument("--output", type=Path, default=ROOT / "phase3_3_representation_audit")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {args.output}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    summary_dir = args.output / "summary"
    summary_dir.mkdir(parents=True)
    distance_rows: list[dict[str, object]] = []
    margin_summary: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    dispersion_rows: list[dict[str, object]] = []
    centroid_rows: list[dict[str, object]] = []
    centroid_distance_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    evidence: dict[str, object] = {"training_performed": False, "models": {}}

    for method, source in METHODS.items():
        checkpoint = source["checkpoint"]
        prediction_path = source["predictions"]
        if not checkpoint.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"frozen {method} artifact missing: checkpoint={checkpoint}, predictions={prediction_path}")
        model = load_model(checkpoint, device)
        reference_ids, reference_labels, reference_logits = prediction_reference(prediction_path)
        replay_errors: dict[str, float] = {}
        for transform_name in source["transform_candidates"]:
            candidate_test = extract_split(
                model,
                split_loaders(args.data_root, args.batch_size, make_transform(transform_name))["test"],
                device,
            )
            if not (np.array_equal(candidate_test["sample_id"], reference_ids) and np.array_equal(candidate_test["labels"], reference_labels)):
                raise ValueError(f"{method} test IDs or labels do not reproduce saved artifact")
            replay_errors[transform_name] = float(np.abs(candidate_test["logits"] - reference_logits).max())
        matches = [name for name, error in replay_errors.items() if error <= 1e-4]
        if len(matches) != 1:
            raise ValueError(f"{method} preprocessing replay is not uniquely verified: {replay_errors}")
        transform_name = matches[0]
        maximum_logit_error = replay_errors[transform_name]
        splits = {
            name: extract_split(model, loader, device)
            for name, loader in split_loaders(args.data_root, args.batch_size, make_transform(transform_name)).items()
        }

        output = args.output / method / "seed_0"
        output.mkdir(parents=True)
        np.savez_compressed(
            output / "features.npz",
            **{f"{split}_{key}": value for split, values in splits.items() for key, value in values.items()},
        )

        train = splits["train"]
        raw_centroids = class_centroids(train["features"], train["labels"], 5)
        train_unit = l2_normalize(train["features"])
        normalized_centroids = class_centroids(train_unit, train["labels"], 5)
        raw_centroid_distances = euclidean_distances(raw_centroids, raw_centroids)
        cosine_centroid_distances = cosine_distances(normalized_centroids, normalized_centroids)
        for space, centroids, centroid_distances in (
            ("raw_euclidean", raw_centroids, raw_centroid_distances),
            ("l2_normalized_cosine", normalized_centroids, cosine_centroid_distances),
        ):
            for class_index, centroid in enumerate(centroids):
                centroid_rows.append(
                    {"method": method, "space": space, "class": class_index, "feature_dimension": int(centroid.size), "l2_norm": float(np.linalg.norm(centroid))}
                )
            for source_class in range(5):
                for target_class in range(5):
                    centroid_distance_rows.append(
                        {"method": method, "space": space, "source_class": source_class, "target_class": target_class, "distance": float(centroid_distances[source_class, target_class])}
                    )

        for split_name, values in splits.items():
            raw_distances = euclidean_distances(values["features"], raw_centroids)
            normalized_distances = cosine_distances(l2_normalize(values["features"]), normalized_centroids)
            for space, distances in (("raw_euclidean", raw_distances), ("l2_normalized_cosine", normalized_distances)):
                nearest = nearest_centroid(distances)
                for true_class, references in ((4, (3, 2)), (0, (1,))):
                    distance_rows.extend(summarize_distances(method, space, split_name, values["labels"], distances, true_class))
                    margin_summary.extend(margin_rows(method, space, split_name, values["labels"], distances, true_class, references))
                    mask = values["labels"] == true_class
                    for predicted_class in range(5):
                        routing_rows.append(
                            {"method": method, "space": space, "split": split_name, "true_class": true_class, "predicted_class": predicted_class, "count": int(((nearest == predicted_class) & mask).sum()), "fraction": float((nearest[mask] == predicted_class).mean())}
                        )
                metric_rows.append({"method": method, "space": space, "split": split_name, **decision_metrics(values["labels"], nearest)})
                if split_name == "train":
                    reference_features = values["features"] if space == "raw_euclidean" else l2_normalize(values["features"])
                    reference_centroids = raw_centroids if space == "raw_euclidean" else normalized_centroids
                    dispersion = within_class_dispersion(reference_features, values["labels"], reference_centroids)
                    for class_index in range(5):
                        dispersion_rows.append(
                            {"method": method, "space": space, "class": class_index, **{name: float(values_[class_index]) if name != "count" else int(values_[class_index]) for name, values_ in dispersion.items()}}
                        )
                if split_name in {"val", "test"}:
                    per_sample = []
                    for index in range(values["labels"].size):
                        per_sample.append(
                            {
                                "sample_id": int(values["sample_id"][index]),
                                "true_label": int(values["labels"][index]),
                                "mode_decision": int(values["mode_decision"][index]),
                                "l1_decision": int(values["l1_bayes_decision"][index]),
                                "nearest_centroid": int(nearest[index]),
                                **{f"distance_to_{class_index}": float(distances[index, class_index]) for class_index in range(5)},
                            }
                        )
                    write_csv(output / f"{split_name}_{space}_per_sample.csv", per_sample)
                    for true_class in (0, 4):
                        mask = values["labels"] == true_class
                        for feature_prediction in range(5):
                            for head_name, head_prediction in (("mode", values["mode_decision"]), ("l1", values["l1_bayes_decision"])):
                                for head_class in range(5):
                                    count = int((mask & (nearest == feature_prediction) & (head_prediction == head_class)).sum())
                                    if count:
                                        comparison_rows.append(
                                            {"method": method, "space": space, "split": split_name, "true_class": true_class, "nearest_centroid": feature_prediction, "head_decision": head_name, "head_prediction": head_class, "count": count}
                                        )

        evidence["models"][method] = {
            "checkpoint": str(checkpoint),
            "saved_predictions": str(prediction_path),
            "preprocessing": transform_name,
            "preprocessing_replay_max_logit_error": replay_errors,
            "feature_definition": "input to model.fc after ResNet18 avgpool and flatten",
            "feature_dimension": int(train["features"].shape[1]),
            "split_counts": {split: int(values["labels"].size) for split, values in splits.items()},
            "max_saved_test_logit_error": maximum_logit_error,
        }

    write_csv(summary_dir / "centroids.csv", centroid_rows)
    write_csv(summary_dir / "centroid_distance_matrices.csv", centroid_distance_rows)
    write_csv(summary_dir / "distance_summary.csv", distance_rows)
    write_csv(summary_dir / "distance_margins.csv", margin_summary)
    write_csv(summary_dir / "nearest_centroid_routing.csv", routing_rows)
    write_csv(summary_dir / "nearest_centroid_metrics.csv", metric_rows)
    write_csv(summary_dir / "within_class_dispersion.csv", dispersion_rows)
    write_csv(summary_dir / "representation_head_comparison.csv", comparison_rows)
    (summary_dir / "phase3_3_summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
