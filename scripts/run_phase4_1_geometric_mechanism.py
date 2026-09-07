#!/usr/bin/env python3
"""Frozen A/C angular-geometry audit for Phase 4.1; no fitting is performed."""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("outputs/phase4_1_geometric_mechanism")
BOOTSTRAP_SEED = 4101
BOOTSTRAP_REPS = 10_000


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError("non-finite or zero-norm vectors")
    return values / norms


def load_prediction_decisions(path: Path, ids: np.ndarray, field: str) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {str(row["sample_id"]): int(row[field]) for row in rows}
    if len(values) != len(ids) or set(values) != {str(value) for value in ids}:
        raise ValueError(f"prediction ID mismatch for {path}")
    return np.asarray([values[str(value)] for value in ids], dtype=np.int64)


def state_direction(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "weight" in payload:
        weight = payload["weight"].detach().cpu().numpy()
    else:
        state = payload.get("head_state_dict", payload.get("state_dict"))
        if state is None or "direction" not in state:
            raise ValueError(f"no direction state in {path}")
        weight = state["direction"].detach().cpu().numpy()
    return unit_rows(weight.astype(np.float64))


def original_direction(checkpoint: Path) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    return unit_rows(state["fc.weight"].detach().cpu().numpy().astype(np.float64))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q10": float(np.quantile(values, .10)), "q25": float(np.quantile(values, .25)),
        "q75": float(np.quantile(values, .75)), "q90": float(np.quantile(values, .90)),
        "fraction_positive": float(np.mean(values > 0)),
    }


def bootstrap_median(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) < 10:
        return {"n": int(len(values)), "seed": BOOTSTRAP_SEED, "reps": 0, "median": float(np.median(values)) if len(values) else None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    medians = np.median(values[samples], axis=1)
    return {"n": int(len(values)), "seed": BOOTSTRAP_SEED, "reps": BOOTSTRAP_REPS, "median": float(np.median(values)), "ci95_low": float(np.quantile(medians, .025)), "ci95_high": float(np.quantile(medians, .975))}


def rotation_rows(a: np.ndarray, c: np.ndarray) -> list[dict[str, float | int]]:
    cosine = np.sum(a * c, axis=1).clip(-1, 1)
    return [{"class": int(k), "cosine_a_c": float(cosine[k]), "rotation_degrees": float(np.degrees(np.arccos(cosine[k])))} for k in range(5)]


def centroid_strata(features: np.ndarray, labels: np.ndarray, centers: np.ndarray, rare: int) -> np.ndarray:
    distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    result = np.full(len(labels), "not_rare", dtype=object)
    rare_mask = labels == rare
    result[rare_mask & (nearest == rare)] = "rare_like"
    result[rare_mask & (nearest != rare)] = "collapsed"
    return result


def analysis(features: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray, decisions_a: np.ndarray, decisions_c: np.ndarray, directions_a: np.ndarray, directions_c: np.ndarray, dataset: str, strata: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    if features.shape != (len(labels), 512) or directions_a.shape != (5, 512) or directions_c.shape != (5, 512):
        raise ValueError("unexpected feature/direction shapes")
    rare, adjacent, deep = 4, 3, 2
    normalized = unit_rows(features.astype(np.float64))
    score_a, score_c = normalized @ directions_a.T, normalized @ directions_c.T
    margin_a, margin_c = score_a[:, rare] - score_a[:, adjacent], score_c[:, rare] - score_c[:, adjacent]
    deep_a, deep_c = score_a[:, rare] - score_a[:, deep], score_c[:, rare] - score_c[:, deep]
    rare_mask = labels == rare
    inward_a, inward_c = decisions_a != rare, decisions_c != rare
    groups = {
        "recovered": rare_mask & inward_a & (decisions_c == rare),
        "still_inward": rare_mask & inward_a & inward_c,
        "lost": rare_mask & (decisions_a == rare) & inward_c,
    }
    rows = []
    for index in np.flatnonzero(rare_mask):
        group = next((name for name, mask in groups.items() if mask[index]), "unchanged_correct")
        rows.append({"dataset": dataset, "sample_id": str(sample_ids[index]), "label": int(labels[index]), "decision_a": int(decisions_a[index]), "decision_c": int(decisions_c[index]), "inward_distance_a": int(rare - decisions_a[index]), "inward_distance_c": int(rare - decisions_c[index]), "group": group, "angular_margin_a": float(margin_a[index]), "angular_margin_c": float(margin_c[index]), "delta_angular_margin": float(margin_c[index] - margin_a[index]), "angular_margin_deep_a": float(deep_a[index]), "angular_margin_deep_c": float(deep_c[index]), "rare_alignment_a": float(score_a[index, rare]), "rare_alignment_c": float(score_c[index, rare]), "delta_rare_alignment": float(score_c[index, rare] - score_a[index, rare]), "adjacent_alignment_a": float(score_a[index, adjacent]), "adjacent_alignment_c": float(score_c[index, adjacent]), "delta_adjacent_alignment": float(score_c[index, adjacent] - score_a[index, adjacent]), "representation_stratum": str(strata[index]) if strata is not None else "not_available"})
    rare_unit_mean = unit_rows(normalized[rare_mask].sum(axis=0, keepdims=True))[0]
    groups_summary = {}
    for name, mask in groups.items():
        delta = margin_c[mask] - margin_a[mask]
        groups_summary[name] = {"count": int(mask.sum()), "margin_a": summarize(margin_a[mask]) if mask.any() else None, "margin_c": summarize(margin_c[mask]) if mask.any() else None, "delta_margin": summarize(delta) if mask.any() else None, "delta_margin_bootstrap_median": bootstrap_median(delta), "delta_rare_alignment": summarize(score_c[mask, rare] - score_a[mask, rare]) if mask.any() else None, "delta_adjacent_alignment": summarize(score_c[mask, adjacent] - score_a[mask, adjacent]) if mask.any() else None}
    stratum_summary = {}
    if strata is not None:
        for name in ("rare_like", "collapsed"):
            mask = rare_mask & (strata == name)
            delta = margin_c[mask] - margin_a[mask]
            stratum_summary[name] = {"count": int(mask.sum()), "recovered": int((groups["recovered"] & mask).sum()), "still_inward": int((groups["still_inward"] & mask).sum()), "margin_a": summarize(margin_a[mask]) if mask.any() else None, "margin_c": summarize(margin_c[mask]) if mask.any() else None, "delta_margin": summarize(delta) if mask.any() else None}
    corr = lambda x, y: float(spearmanr(x, y).statistic) if len(x) > 2 and np.unique(x).size > 1 else None
    summary = {"dataset": dataset, "feature_shape": list(features.shape), "rare_class": rare, "rare_support": int(rare_mask.sum()), "angular_definition": "unit_feature dot unit_classifier_direction; margin is rare minus adjacent", "angular_margin_a": summarize(margin_a[rare_mask]), "angular_margin_c": summarize(margin_c[rare_mask]), "angular_margin_change": summarize((margin_c - margin_a)[rare_mask]), "rare_alignment_a": summarize(score_a[rare_mask, rare]), "rare_alignment_c": summarize(score_c[rare_mask, rare]), "adjacent_alignment_a": summarize(score_a[rare_mask, adjacent]), "adjacent_alignment_c": summarize(score_c[rare_mask, adjacent]), "deep_margin_a": summarize(deep_a[rare_mask]), "deep_margin_c": summarize(deep_c[rare_mask]), "normalized_rare_beats_adjacent_a": float((score_a[rare_mask, rare] > score_a[rare_mask, adjacent]).mean()), "normalized_rare_beats_adjacent_c": float((score_c[rare_mask, rare] > score_c[rare_mask, adjacent]).mean()), "margin_vs_inward_distance_spearman_a": corr(margin_a[rare_mask], rare - decisions_a[rare_mask]), "margin_vs_inward_distance_spearman_c": corr(margin_c[rare_mask], rare - decisions_c[rare_mask]), "direction_rotation": rotation_rows(directions_a, directions_c), "mean_rare_feature_alignment": {"rare_direction_a": float(directions_a[rare] @ rare_unit_mean), "rare_direction_c": float(directions_c[rare] @ rare_unit_mean), "adjacent_direction_a": float(directions_a[adjacent] @ rare_unit_mean), "adjacent_direction_c": float(directions_c[adjacent] @ rare_unit_mean)}, "groups": groups_summary, "representation_strata": stratum_summary}
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def run_retina(out: Path) -> None:
    root = Path("outputs/retinamnist")
    frozen = np.load(root / "phase3_10a_rop_objective_falsification/frozen_features/train_rps_features.npz")
    features, labels, ids = frozen["features"], frozen["labels"], frozen["sample_id"]
    folds = np.asarray([int(row["fold"]) for row in csv.DictReader((root / "phase3_10a_rop_objective_falsification/fold_assignments/assignments.csv").open())])
    checkpoint = root / "native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt"
    direction_a = original_direction(checkpoint)
    direction_c = np.empty((len(labels), 5, 512), dtype=np.float64)
    for fold in range(5):
        state = state_direction(root / f"phase3_10c_direction_only_head/fold_states/fold_{fold}.pt")
        direction_c[folds == fold] = state
    # analysis accepts a common direction; evaluate folds separately then pool sample rows to preserve OOF heads.
    a_decision = frozen["original_l1_decision"].astype(np.int64)
    c_decision = load_prediction_decisions(root / "phase3_10c_direction_only_head/oof_predictions/predictions.csv", ids, "l1_decision")
    audit = np.load(root / "native28/phase3_3_representation_audit_replay_verified/rps/seed_0/features.npz")
    order = {int(value): index for index, value in enumerate(audit["train_sample_id"])}
    if set(order) != set(map(int, ids)):
        raise ValueError("Retina phase3.3/phase3.10 train ID mismatch")
    centers = np.stack([audit["train_features"][audit["train_labels"] == k].mean(axis=0) for k in range(5)])
    strata = centroid_strata(features, labels, centers, 4)
    all_rows, fold_summaries = [], []
    for fold in range(5):
        mask = folds == fold
        summary, rows = analysis(features[mask], labels[mask], ids[mask], a_decision[mask], c_decision[mask], direction_a, direction_c[mask][0], "retinamnist", strata[mask])
        summary["fold"] = fold; fold_summaries.append(summary); all_rows.extend(rows)
    # recompute pooled fields from the sample-level rows and directions to avoid treating a direction average as a head.
    aggregate = {"dataset": "retinamnist", "protocol": "pooled 5-fold OOF; each sample uses its held-out C fold state", "feature_shape": list(features.shape), "rare_support": int((labels == 4).sum()), "fold_summaries": fold_summaries, "direction_rotation_by_fold": [{"fold": fold, "rows": rotation_rows(direction_a, direction_c[folds == fold][0])} for fold in range(5)]}
    pooled = pooled_from_rows(all_rows)
    aggregate.update(pooled); aggregate["representation_strata"] = stratum_from_rows(all_rows)
    finalize(out, aggregate, all_rows, "retinamnist")


def pooled_from_rows(rows: list[dict]) -> dict:
    arr = lambda key, group=None: np.asarray([row[key] for row in rows if group is None or row["group"] == group], dtype=float)
    a, c = arr("angular_margin_a"), arr("angular_margin_c")
    output = {"angular_margin_a": summarize(a), "angular_margin_c": summarize(c), "angular_margin_change": summarize(c - a), "normalized_rare_beats_adjacent_a": float(np.mean(a > 0)), "normalized_rare_beats_adjacent_c": float(np.mean(c > 0)), "groups": {}}
    for group in ("recovered", "still_inward", "lost"):
        delta = arr("delta_angular_margin", group)
        output["groups"][group] = {"count": int(len(delta)), "delta_margin": summarize(delta) if len(delta) else None, "delta_margin_bootstrap_median": bootstrap_median(delta), "delta_rare_alignment": summarize(arr("delta_rare_alignment", group)) if len(delta) else None, "delta_adjacent_alignment": summarize(arr("delta_adjacent_alignment", group)) if len(delta) else None}
    return output


def stratum_from_rows(rows: list[dict]) -> dict:
    out = {}
    for label in ("rare_like", "collapsed"):
        chosen = [row for row in rows if row["representation_stratum"] == label]
        out[label] = {"count": len(chosen), "recovered": sum(row["group"] == "recovered" for row in chosen), "still_inward": sum(row["group"] == "still_inward" for row in chosen), "delta_margin": summarize(np.asarray([row["delta_angular_margin"] for row in chosen])) if chosen else None}
    return out


def run_single(dataset: str, out: Path, feature_path: Path, a_state: Path | None, c_state: Path, a_prediction: Path, c_prediction: Path, decision_field: str, centroid_json: Path | None = None, a_weight: np.ndarray | None = None) -> None:
    cache = np.load(feature_path, allow_pickle=False)
    features, labels, ids = cache["features"], cache["labels"], cache["sample_ids"]
    decision_a = load_prediction_decisions(a_prediction, ids, decision_field) if a_prediction.suffix == ".csv" else np.load(a_prediction, allow_pickle=False)["l1"]
    decision_c = load_prediction_decisions(c_prediction, ids, decision_field) if c_prediction.suffix == ".csv" else np.load(c_prediction, allow_pickle=False)["l1"]
    strata = None
    if centroid_json is not None:
        geometry = json.loads(centroid_json.read_text())
        centers = np.asarray(geometry["rps"]["spaces"]["raw_euclidean"]["centroids"], dtype=np.float64)
        strata = centroid_strata(features, labels, centers, 4)
    if a_state is None and a_weight is None:
        raise ValueError("A direction requires a saved state or frozen original weight")
    direction_a = state_direction(a_state) if a_state is not None else unit_rows(a_weight.astype(np.float64))
    summary, rows = analysis(features, labels, ids, decision_a, decision_c, direction_a, state_direction(c_state), dataset, strata)
    finalize(out, summary, rows, dataset)


def finalize(out: Path, summary: dict, rows: list[dict], dataset: str) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    summary["analysis"] = {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_reps": BOOTSTRAP_REPS, "training_performed": False, "head_fitting_performed": False, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.executable}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out / "rare_sample_geometry.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("retinamnist", "utkface", "solar"))
    parser.add_argument("--solar-feature-root", type=Path)
    parser.add_argument("--solar-centroids", type=Path)
    args = parser.parse_args()
    if args.dataset == "retinamnist":
        run_retina(ROOT / "retinamnist")
    elif args.dataset == "utkface":
        root = Path("outputs/utkface/phase3_13_direction_scale_mechanism_confirmation")
        cache = np.load(root / "frozen_val_features.npz", allow_pickle=False)
        temp = ROOT / "utkface_inputs.npz"; temp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(temp, features=cache["features"], labels=cache["labels"], sample_ids=cache["sample_ids"])
        run_single("utkface", ROOT / "utkface", temp, None, root / "head_states/C.pt", root / "predictions/A.csv", root / "predictions/C.csv", "l1", a_weight=cache["original_weight"])
    else:
        if args.solar_feature_root is None or args.solar_centroids is None:
            parser.error("Solar requires --solar-feature-root and --solar-centroids")
        root = args.solar_feature_root
        run_single("solar", ROOT / "solar", root / "test.npz", root / "head_states/A_original_rps.pt", root / "head_states/C_direction_only.pt", root / "evaluation/A_original_rps/predictions.npz", root / "evaluation/C_direction_only/predictions.npz", "l1", args.solar_centroids)


if __name__ == "__main__":
    main()
