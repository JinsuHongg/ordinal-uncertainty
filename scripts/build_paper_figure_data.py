#!/usr/bin/env python3
"""Build and verify canonical Figure 1--4 data tables from audited artifacts.

This is deterministic data consolidation only. It never trains, evaluates a new
model, or creates a plot. Source paths and extraction constraints are frozen in
``docs/research/figure_source_artifact_audit.md``.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ordinal_uncertainty.metrics.decision import bayes_decisions


ROOT = Path("outputs")
OUT = ROOT / "manuscript" / "figure_data"
TOLERANCE = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(value), dtype=np.float64)


def load_csv_predictions(path: Path, *, label_field: str, decision_field: str | None) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    ids = np.asarray([str(row["sample_id"]) for row in rows])
    labels = np.asarray([int(row[label_field]) for row in rows])
    logits = np.stack([parse_vector(row["logits"]) for row in rows])
    probabilities = np.stack([parse_vector(row["probabilities"]) for row in rows])
    derived = bayes_decisions(probabilities)["l1_bayes_decision"]
    if decision_field is not None:
        saved = np.asarray([int(row[decision_field]) for row in rows])
        if not np.array_equal(derived, saved):
            raise ValueError(f"saved and recomputed L1 actions disagree: {path}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"non-unique IDs: {path}")
    return {"ids": ids, "labels": labels, "logits": logits, "probabilities": probabilities, "l1": derived}


def load_npz_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", "labels", "logits", "probabilities", "l1"}
        if missing := required - set(archive.files):
            raise ValueError(f"{path} missing {sorted(missing)}")
        result = {"ids": archive["sample_ids"].astype(str), "labels": archive["labels"].astype(int),
                  "logits": archive["logits"].astype(np.float64),
                  "probabilities": archive["probabilities"].astype(np.float64), "l1": archive["l1"].astype(int)}
    if len(np.unique(result["ids"])) != len(result["ids"]):
        raise ValueError(f"non-unique IDs: {path}")
    derived = bayes_decisions(result["probabilities"])["l1_bayes_decision"]
    if not np.array_equal(derived, result["l1"]):
        raise ValueError(f"saved and recomputed L1 actions disagree: {path}")
    return result


def assert_aligned(left: dict[str, np.ndarray], right: dict[str, np.ndarray], name: str) -> None:
    left_index = {sample_id: i for i, sample_id in enumerate(left["ids"])}
    right_index = {sample_id: i for i, sample_id in enumerate(right["ids"])}
    if set(left_index) != set(right_index):
        raise ValueError(f"ID set mismatch: {name}")
    for sample_id, left_i in left_index.items():
        if left["labels"][left_i] != right["labels"][right_index[sample_id]]:
            raise ValueError(f"label mismatch for {sample_id}: {name}")


def reorder(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    index = {sample_id: i for i, sample_id in enumerate(candidate["ids"])}
    return {key: values[[index[sample_id] for sample_id in reference["ids"]]] for key, values in candidate.items()}


def endpoint_rows(dataset: str, population: str, source: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(np.isin(source["labels"], [0, 4])):
        label = int(source["labels"][i])
        probability = source["probabilities"][i]
        mean = float(probability @ np.arange(5))
        rows.append({
            "dataset": dataset, "population": population, "model_objective": "CE", "seed": 0,
            "sample_id": source["ids"][i], "true_label": label, "endpoint": "lower" if label == 0 else "upper",
            "l1_prediction": int(source["l1"][i]), "ordinal_error": abs(label - int(source["l1"][i])),
            "predictive_mean": mean, "inward_shrinkage": mean if label == 0 else 4.0 - mean,
            **{f"p{k}": float(probability[k]) for k in range(5)},
        })
    return rows


def endpoint_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for endpoint in ("lower", "upper"):
            selected = [row for row in rows if row["dataset"] == dataset and row["endpoint"] == endpoint]
            if not selected:
                continue
            result.append({
                "dataset": dataset, "model_objective": "CE", "seed": 0, "endpoint": endpoint,
                "true_label": selected[0]["true_label"], "support": len(selected),
                "routing_0": sum(row["l1_prediction"] == 0 for row in selected),
                "routing_1": sum(row["l1_prediction"] == 1 for row in selected),
                "routing_2": sum(row["l1_prediction"] == 2 for row in selected),
                "routing_3": sum(row["l1_prediction"] == 3 for row in selected),
                "routing_4": sum(row["l1_prediction"] == 4 for row in selected),
                "l1_mae": float(np.mean([row["ordinal_error"] for row in selected])),
                "predictive_mean": float(np.mean([row["predictive_mean"] for row in selected])),
                "inward_shrinkage": float(np.mean([row["inward_shrinkage"] for row in selected])),
            })
    return result


def build_figure1() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retina = load_csv_predictions(ROOT / "retinamnist/resolution_sanity_check/seed_0/size_28/predictions.csv", label_field="true_label", decision_field=None)
    utk = load_csv_predictions(ROOT / "utkface/phase3_7a_failure_replication/ce/seed_0/predictions.csv", label_field="true_label", decision_field="l1_bayes_decision")
    solar = load_npz_predictions(ROOT / "solar/phase3_8_shrinkage_confirmation/ce/seed_0/evaluation/predictions.npz")
    samples = endpoint_rows("RetinaMNIST", "original_test", retina)
    samples += endpoint_rows("UTKFace", "original_test", utk)
    samples += endpoint_rows("Solar", "retained_aligned_future_test", solar)
    return samples, endpoint_summary(samples)


def raw_centroid_groups(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centroids = np.stack([features[labels == item].mean(axis=0) for item in range(5)])
    return ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def load_retina_ce_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / "retinamnist/native28/phase3_3_representation_audit_replay_verified/ce/seed_0/features.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["train_sample_id"].astype(str), archive["train_labels"].astype(int), archive["train_features"].astype(np.float64)


def load_solar_features(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / f"solar/phase3_9_mechanism_audit/features/ce/{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["sample_ids"].astype(str), archive["labels"].astype(int), archive["features"].astype(np.float64)


def routing(values: np.ndarray) -> dict[str, int]:
    return {f"routing_{item}": int((values == item).sum()) for item in range(5)}


def mechanism_rows(dataset: str, population: str, ids: np.ndarray, labels: np.ndarray, nearest: np.ndarray,
                   a: dict[str, np.ndarray], c: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    a = reorder({"ids": ids, "labels": labels}, a)
    c = reorder({"ids": ids, "labels": labels}, c)
    rare = labels == 4
    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(rare):
        a_mu = float(a["probabilities"][i] @ np.arange(5))
        c_mu = float(c["probabilities"][i] @ np.arange(5))
        group = "rare_end_like_raw_nearest_4" if nearest[i] == 4 else "representation_inward_raw_nearest_interior"
        rows.append({
            "dataset": dataset, "population": population, "representation_objective": "CE", "seed": 0,
            "sample_id": ids[i], "true_label": 4, "nearest_centroid": int(nearest[i]), "representation_group": group,
            "a_l1_prediction": int(a["l1"][i]), "c_l1_prediction": int(c["l1"][i]),
            "a_exact": int(a["l1"][i] == 4), "c_exact": int(c["l1"][i] == 4),
            "a_to_c_exact_recovery": int(a["l1"][i] != 4 and c["l1"][i] == 4),
            "a_predictive_mean": a_mu, "c_predictive_mean": c_mu, "delta_predictive_mean": c_mu - a_mu,
            "a_inward_shrinkage": 4.0 - a_mu, "c_inward_shrinkage": 4.0 - c_mu,
            "delta_inward_shrinkage": a_mu - c_mu,
            "a_p4": float(a["probabilities"][i, 4]), "c_p4": float(c["probabilities"][i, 4]),
        })
    return rows


def mechanism_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for group in sorted({row["representation_group"] for row in rows if row["dataset"] == dataset}):
            selected = [row for row in rows if row["dataset"] == dataset and row["representation_group"] == group]
            result.append({
                "dataset": dataset, "representation_objective": "CE", "seed": 0, "representation_group": group,
                "support": len(selected), "nearest_centroid": 4 if group.startswith("rare_end") else "interior",
                **routing(np.asarray([row["a_l1_prediction"] for row in selected])),
                **{f"c_{key}": value for key, value in routing(np.asarray([row["c_l1_prediction"] for row in selected])).items()},
                "a_exact": sum(row["a_exact"] for row in selected), "c_exact": sum(row["c_exact"] for row in selected),
                "a_to_c_exact_recovery": sum(row["a_to_c_exact_recovery"] for row in selected),
                "a_l1_mae": float(np.mean([abs(4 - row["a_l1_prediction"]) for row in selected])),
                "c_l1_mae": float(np.mean([abs(4 - row["c_l1_prediction"]) for row in selected])),
                "mean_delta_predictive_mean": float(np.mean([row["delta_predictive_mean"] for row in selected])),
                "mean_delta_inward_shrinkage": float(np.mean([row["delta_inward_shrinkage"] for row in selected])),
            })
    return result


def build_figure2() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retina_ids, retina_labels, retina_features = load_retina_ce_features()
    retina_nearest = raw_centroid_groups(retina_features, retina_labels)
    retina_a = load_csv_predictions(ROOT / "retinamnist/phase3_18a_ce_direction_robustness/predictions/A_CE_original.csv", label_field="label", decision_field="l1_decision")
    retina_c = load_csv_predictions(ROOT / "retinamnist/phase3_18a_ce_direction_robustness/predictions/C_CE_direction_only.csv", label_field="label", decision_field="l1_decision")
    assert_aligned({"ids": retina_ids, "labels": retina_labels}, retina_a, "Retina feature/A")
    assert_aligned(retina_a, retina_c, "Retina A/C")

    solar_train_ids, solar_train_labels, solar_train_features = load_solar_features("train")
    solar_ids, solar_labels, solar_features = load_solar_features("test")
    if len(np.unique(solar_train_ids)) != len(solar_train_ids):
        raise ValueError("non-unique Solar train IDs")
    # Recompute on the train partition only, as required by the audit.
    centroids = np.stack([solar_train_features[solar_train_labels == k].mean(axis=0) for k in range(5)])
    solar_nearest = ((solar_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    solar_a = load_npz_predictions(ROOT / "solar/phase3_18b_ce_direction_robustness/evaluation/A_original_ce/predictions.npz")
    solar_c = load_npz_predictions(ROOT / "solar/phase3_18b_ce_direction_robustness/evaluation/C_direction_only/predictions.npz")
    assert_aligned({"ids": solar_ids, "labels": solar_labels}, solar_a, "Solar feature/A")
    assert_aligned(solar_a, solar_c, "Solar A/C")

    samples = mechanism_rows("RetinaMNIST", "training_only_oof", retina_ids, retina_labels, retina_nearest, retina_a, retina_c)
    samples += mechanism_rows("Solar", "retained_aligned_future_test", solar_ids, solar_labels, solar_nearest, solar_a, solar_c)
    return samples, mechanism_summary(samples)


def direction_summary_row(dataset: str, representation: str, population: str, condition: str, source: dict[str, np.ndarray]) -> dict[str, Any]:
    rare = source["labels"] == 4
    errors = np.abs(source["l1"][rare] - 4)
    probabilities = source["probabilities"][rare]
    logits = source["logits"][rare]
    predictive_mean = probabilities @ np.arange(5)
    return {
        "dataset": dataset, "representation_objective": representation, "population": population, "seed": 0,
        "condition": condition, "rare_class": 4, "rare_n": int(rare.sum()),
        "rare_l1_mae": float(errors.mean()), "rare_exact_count": int((source["l1"][rare] == 4).sum()),
        "rare_exact_fraction": float((source["l1"][rare] == 4).mean()),
        "rare_predictive_mean": float(predictive_mean.mean()), "rare_inward_shrinkage": float(4.0 - predictive_mean.mean()),
        "rare_mean_p4": float(probabilities[:, 4].mean()), "rare_severe_fraction": float((errors >= 2).mean()),
        "rare_z4_z3_mean": float((logits[:, 4] - logits[:, 3]).mean()),
        "rare_z4_z3_median": float(np.median(logits[:, 4] - logits[:, 3])),
        "rare_z4_z3_positive_fraction": float((logits[:, 4] > logits[:, 3]).mean()),
        "global_l1_mae": float(np.abs(source["l1"] - source["labels"]).mean()),
    }


def build_figure3() -> list[dict[str, Any]]:
    retina_rps_a = load_csv_predictions(ROOT / "retinamnist/phase3_10a_rop_objective_falsification/oof_predictions/A_original_rps.csv", label_field="label", decision_field="l1_decision")
    retina_rps_c = load_csv_predictions(ROOT / "retinamnist/phase3_10c_direction_only_head/oof_predictions/predictions.csv", label_field="label", decision_field="l1_decision")
    retina_ce_a = load_csv_predictions(ROOT / "retinamnist/phase3_18a_ce_direction_robustness/predictions/A_CE_original.csv", label_field="label", decision_field="l1_decision")
    retina_ce_c = load_csv_predictions(ROOT / "retinamnist/phase3_18a_ce_direction_robustness/predictions/C_CE_direction_only.csv", label_field="label", decision_field="l1_decision")
    solar_rps_a = load_npz_predictions(ROOT / "solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24/evaluation/A_original_rps/predictions.npz")
    solar_rps_c = load_npz_predictions(ROOT / "solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24/evaluation/C_direction_only/predictions.npz")
    solar_ce_a = load_npz_predictions(ROOT / "solar/phase3_18b_ce_direction_robustness/evaluation/A_original_ce/predictions.npz")
    solar_ce_c = load_npz_predictions(ROOT / "solar/phase3_18b_ce_direction_robustness/evaluation/C_direction_only/predictions.npz")
    for name, a, c in (("Retina RPS", retina_rps_a, retina_rps_c), ("Retina CE", retina_ce_a, retina_ce_c),
                       ("Solar RPS", solar_rps_a, solar_rps_c), ("Solar CE", solar_ce_a, solar_ce_c)):
        assert_aligned(a, c, name)
    return [
        direction_summary_row("RetinaMNIST", "RPS", "training_only_oof", "A_original", retina_rps_a),
        direction_summary_row("RetinaMNIST", "RPS", "training_only_oof", "C_direction_only", retina_rps_c),
        direction_summary_row("RetinaMNIST", "CE", "training_only_oof", "A_original", retina_ce_a),
        direction_summary_row("RetinaMNIST", "CE", "training_only_oof", "C_direction_only", retina_ce_c),
        direction_summary_row("Solar", "RPS", "retained_aligned_future_test", "A_original", solar_rps_a),
        direction_summary_row("Solar", "RPS", "retained_aligned_future_test", "C_direction_only", solar_rps_c),
        direction_summary_row("Solar", "CE", "retained_aligned_future_test", "A_original", solar_ce_a),
        direction_summary_row("Solar", "CE", "retained_aligned_future_test", "C_direction_only", solar_ce_c),
    ]


def build_figure4() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase19 = ROOT / "retinamnist/phase3_19_sampling_objective_direction_disentanglement"
    predictions = read_csv(phase19 / "predictions/oof_predictions.csv")
    margins = {(row["condition"], row["target"], row["margin"]): row for row in read_csv(phase19 / "metrics/margins.csv")}
    cosine_rows = read_csv(phase19 / "parameters/direction_cosines.csv")
    condition_info = {
        "C_balanced_ce": ("balanced", "CE"), "E_natural_ce": ("natural", "CE"),
        "F_balanced_rps": ("balanced", "RPS"), "G_natural_rps": ("natural", "RPS"),
    }
    factorial = []
    for condition, (sampling, objective) in condition_info.items():
        selected = [row for row in predictions if row["condition"] == condition and int(row["label"]) == 4]
        if len(selected) != 66:
            raise ValueError(f"unexpected C4 support for {condition}: {len(selected)}")
        probs = np.stack([parse_vector(row["probabilities"]) for row in selected])
        decisions = np.asarray([int(row["l1_decision"]) for row in selected])
        cosine = [float(row["cosine_vs_original"]) for row in cosine_rows if row["condition"] == condition and int(row["class"]) == 4]
        margin = margins[(condition, "4", "z4-z3")]
        factorial.append({
            "condition": condition, "sampling": sampling, "objective": objective, "representation_objective": "RPS",
            "population": "training_only_oof", "backbone_seed": 0, "rare_n": len(selected),
            "c4_l1_mae": float(np.abs(4 - decisions).mean()), "c4_exact_count": int((decisions == 4).sum()),
            "c4_exact_fraction": float((decisions == 4).mean()), "c4_predictive_mean": float((probs @ np.arange(5)).mean()),
            "c4_shrinkage": float(4 - (probs @ np.arange(5)).mean()), "p4_mean": float(probs[:, 4].mean()),
            "z4_z3_mean": float(margin["mean"]), "z4_z3_median": float(margin["median"]),
            "z4_z3_positive_fraction": float(margin["fraction_positive"]), "class4_direction_cosine_mean": float(np.mean(cosine)),
        })

    phase20 = ROOT / "retinamnist/phase3_20a_imbalance_severity_dose_response/analysis"
    per_seed = [{"n4_support": int(row.pop("n4")), "seed": int(row.pop("seed")), **{key: float(value) for key, value in row.items()}} for row in read_csv(phase20 / "per_seed_severity_metrics.csv")]
    summary = []
    for row in read_csv(phase20 / "severity_mean_std.csv"):
        output = {"n4_support": int(row.pop("n4"))}
        output.update({key: float(value) for key, value in row.items()})
        summary.append(output)
    if len({(row["seed"], row["n4_support"]) for row in per_seed}) != 25 or len(per_seed) != 25:
        raise ValueError("Phase 3.20A is not a complete unique 25-cell grid")
    return factorial, per_seed, summary


def close(actual: float, expected: float, description: str, checks: list[dict[str, Any]]) -> None:
    passed = bool(np.isclose(actual, expected, rtol=0.0, atol=TOLERANCE))
    checks.append({"description": description, "actual": actual, "expected_phase_aggregate": expected, "passed": passed})
    if not passed:
        raise ValueError(f"verification failed: {description}: {actual} != {expected}")


def verify(fig1: list[dict[str, Any]], fig2: list[dict[str, Any]], fig3: list[dict[str, Any]], fig4a: list[dict[str, Any]], fig4b: list[dict[str, Any]], fig4c: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    # Phase-note aggregate targets: 3.7A, 3.8, 3.18A/B, 3.10C, 3.15, 3.19, 3.20A.
    lookup1 = {(row["dataset"], row["endpoint"]): row for row in fig1}
    close(lookup1[("UTKFace", "upper")]["l1_mae"], 0.7313432835820896, "Fig1 UTK CE C4 L1 MAE (Phase 3.7A)", checks)
    close(lookup1[("Solar", "upper")]["l1_mae"], 1.1150922909880565, "Fig1 Solar CE X L1 MAE (Phase 3.8)", checks)
    close(lookup1[("Solar", "lower")]["l1_mae"], 0.10408781226343679, "Fig1 Solar CE class-0 L1 MAE (Phase 3.8)", checks)

    group = {(row["dataset"], row["representation_group"]): row for row in fig2}
    close(float(group[("RetinaMNIST", "rare_end_like_raw_nearest_4")]["support"]), 48, "Fig2 Retina nearest-4 support (Phase 3.18A)", checks)
    close(float(group[("RetinaMNIST", "representation_inward_raw_nearest_interior")]["support"]), 18, "Fig2 Retina inward support (Phase 3.18A)", checks)
    close(float(group[("Solar", "rare_end_like_raw_nearest_4")]["support"]), 724, "Fig2 Solar X-like support (Phase 3.18B)", checks)
    close(float(group[("Solar", "representation_inward_raw_nearest_interior")]["support"]), 197, "Fig2 Solar inward support (Phase 3.18B)", checks)
    close(float(group[("Solar", "rare_end_like_raw_nearest_4")]["a_to_c_exact_recovery"]), 559, "Fig2 Solar X-like exact recovery (Phase 3.18B)", checks)

    f3 = {(row["dataset"], row["representation_objective"], row["condition"]): row for row in fig3}
    targets = {
        ("RetinaMNIST", "RPS", "A_original"): (1.696969696969697, 0, 1.7966842498092026),
        ("RetinaMNIST", "RPS", "C_direction_only"): (1.3484848484848484, 2, 1.38397371227091),
        ("RetinaMNIST", "CE", "A_original"): (1.4393939393939394, 0, 1.677636283873157),
        ("RetinaMNIST", "CE", "C_direction_only"): (1.0303030303030303, 11, 1.244079859306415),
    }
    for key, (mae, exact, shrinkage) in targets.items():
        close(f3[key]["rare_l1_mae"], mae, f"Fig3 {key} rare MAE", checks)
        close(float(f3[key]["rare_exact_count"]), exact, f"Fig3 {key} exact count", checks)
        close(f3[key]["rare_inward_shrinkage"], shrinkage, f"Fig3 {key} shrinkage", checks)
    for key, (mae, exact, shrinkage) in {
        ("Solar", "RPS", "A_original"): (1.1965255157437569, 0, 1.2281334887264839),
        ("Solar", "RPS", "C_direction_only"): (0.6753528773072747, 496, 0.772815235232642),
        ("Solar", "CE", "A_original"): (1.1150922909880565, 0, 1.1360529907603512),
        ("Solar", "CE", "C_direction_only"): (0.5863192182410424, 559, 0.6564395724178245),
    }.items():
        close(f3[key]["rare_l1_mae"], mae, f"Fig3 {key} rare MAE", checks)
        close(float(f3[key]["rare_exact_count"]), exact, f"Fig3 {key} exact count", checks)
        close(f3[key]["rare_inward_shrinkage"], shrinkage, f"Fig3 {key} shrinkage", checks)

    f4 = {row["condition"]: row for row in fig4a}
    for condition, expected_mae, expected_margin in (("C_balanced_ce", 1.3484848484848484, 0.029694214463233948), ("E_natural_ce", 1.696969696969697, -0.9124008417129517), ("F_balanced_rps", 1.3181818181818181, 0.08253799378871918), ("G_natural_rps", 1.7121212121212122, -0.8777045011520386)):
        close(f4[condition]["c4_l1_mae"], expected_mae, f"Fig4 {condition} C4 MAE (Phase 3.19)", checks)
        close(f4[condition]["z4_z3_mean"], expected_margin, f"Fig4 {condition} margin (Phase 3.19)", checks)
    summary = {row["n4_support"]: row for row in fig4c}
    for support in (66, 50, 33, 16, 8):
        rows = [row for row in fig4b if row["n4_support"] == support]
        for raw, aggregate in (("mean_p4", "mean_p4_mean"), ("margin_z4_z3", "margin_z4_z3_mean"), ("shrinkage", "shrinkage_mean")):
            close(float(np.mean([row[raw] for row in rows])), summary[support][aggregate], f"Fig4 N4={support} {raw} saved aggregation (Phase 3.20A)", checks)
    return {"status": "PASS", "tolerance": TOLERANCE, "checks": checks, "check_count": len(checks), "all_passed": all(check["passed"] for check in checks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing canonical tables: {args.output}")
    fig1_samples, fig1_summary = build_figure1()
    fig2_samples, fig2_summary = build_figure2()
    fig3_summary = build_figure3()
    fig4_factorial, fig4_per_seed, fig4_summary = build_figure4()
    verification = verify(fig1_summary, fig2_summary, fig3_summary, fig4_factorial, fig4_per_seed, fig4_summary)
    write_csv(args.output / "figure1_endpoint_samples.csv", fig1_samples)
    write_csv(args.output / "figure1_endpoint_summary.csv", fig1_summary)
    write_csv(args.output / "figure2_mechanism_samples.csv", fig2_samples)
    write_csv(args.output / "figure2_mechanism_summary.csv", fig2_summary)
    write_csv(args.output / "figure3_direction_summary.csv", fig3_summary)
    write_csv(args.output / "figure4_factorial_summary.csv", fig4_factorial)
    write_csv(args.output / "figure4_severity_per_seed.csv", fig4_per_seed)
    write_csv(args.output / "figure4_severity_summary.csv", fig4_summary)
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "verification": verification["status"], "checks": verification["check_count"]}))


if __name__ == "__main__":
    main()
