#!/usr/bin/env python3
"""Frozen confirmatory H1/H2a analysis from completed A/C artifacts.

This analysis is read-only with respect to the completed A/C condition outputs:
it neither trains a backbone/head nor modifies any source artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy.stats import t as student_t


INPUT_BASE = Path("outputs/mechanism_replication/ac")
OUTPUT_BASE = Path("outputs/mechanism_replication/analysis")
SEEDS = (1, 2, 3, 4)
CLASS4 = 4
SPECIFICATION = "retina-confirmatory-h1-h2a-v1"


def sample_standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize one seed's rare-end values using its sample SD (ddof=1)."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        raise ValueError("zero or non-finite predictor variance in rare-end population")
    return (values - mean) / sd, mean, sd


def h1_delta(
    a_l1: np.ndarray, c_l1: np.ndarray, labels: np.ndarray, expected_end_n: int = 66
) -> dict[str, float | int]:
    """Compute the frozen H1 effect as C MAE minus A MAE on true class 4."""
    mask = np.asarray(labels) == CLASS4
    if int(mask.sum()) != expected_end_n:
        raise ValueError(f"expected {expected_end_n} class-4 samples, found {int(mask.sum())}")
    a_mae = float(np.abs(np.asarray(a_l1)[mask] - CLASS4).mean())
    c_mae = float(np.abs(np.asarray(c_l1)[mask] - CLASS4).mean())
    return {
        "a_mae": a_mae,
        "c_mae": c_mae,
        "delta_mae_c_minus_a": c_mae - a_mae,
        "a_exact": int((np.asarray(a_l1)[mask] == CLASS4).sum()),
        "c_exact": int((np.asarray(c_l1)[mask] == CLASS4).sum()),
    }


def h2a_vectors(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Y_C, Y_A, g, and m_adj on the true class-4 population only."""
    mask = arrays["labels"] == CLASS4
    return (
        CLASS4 - arrays["c_predictive_mean"][mask],
        CLASS4 - arrays["a_predictive_mean"][mask],
        arrays["generic_centroid_margin"][mask],
        arrays["endpoint_adj_margin"][mask],
    )


def ordinary_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coefficients, fitted values, and HC3 standard errors."""
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residuals = y - fitted
    inverse = np.linalg.inv(x.T @ x)
    leverage = np.sum((x @ inverse) * x, axis=1)
    adjusted = residuals / (1.0 - leverage)
    hc3 = inverse @ (x.T @ (x * adjusted[:, None] ** 2)) @ inverse
    standard_errors = np.sqrt(np.maximum(np.diag(hc3), 0.0))
    return coefficients, fitted, standard_errors


def model_matrix(y_a: np.ndarray, generic: np.ndarray, ordinal: np.ndarray | None = None) -> np.ndarray:
    columns = [np.ones(len(y_a)), y_a, generic]
    if ordinal is not None:
        columns.append(ordinal)
    return np.column_stack(columns)


def loocv_mse(y_c: np.ndarray, y_a: np.ndarray, generic: np.ndarray, ordinal: np.ndarray | None) -> float:
    """Literal LOOCV with predictor standardization fit inside each training fold."""
    predictions = np.empty(len(y_c), dtype=float)
    for held_out in range(len(y_c)):
        train = np.arange(len(y_c)) != held_out
        za_train, a_mean, a_sd = sample_standardize(y_a[train])
        zg_train, g_mean, g_sd = sample_standardize(generic[train])
        if ordinal is None:
            x_train = model_matrix(za_train, zg_train)
            x_held = model_matrix(
                np.asarray([(y_a[held_out] - a_mean) / a_sd]),
                np.asarray([(generic[held_out] - g_mean) / g_sd]),
            )
        else:
            zo_train, o_mean, o_sd = sample_standardize(ordinal[train])
            x_train = model_matrix(za_train, zg_train, zo_train)
            x_held = model_matrix(
                np.asarray([(y_a[held_out] - a_mean) / a_sd]),
                np.asarray([(generic[held_out] - g_mean) / g_sd]),
                np.asarray([(ordinal[held_out] - o_mean) / o_sd]),
            )
        coefficient, _, _ = ordinary_least_squares(x_train, y_c[train])
        predictions[held_out] = float((x_held @ coefficient).item())
    return float(np.mean((y_c - predictions) ** 2))


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Diagnostic Pearson correlation; inputs are restricted to one seed's C4 rows."""
    if np.std(first, ddof=1) == 0 or np.std(second, ddof=1) == 0:
        raise ValueError("zero-variance diagnostic correlation input")
    return float(np.corrcoef(first, second)[0, 1])


def t_interval(values: list[float]) -> dict[str, float | list[float]]:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    sd = float(data.std(ddof=1))
    critical = float(student_t.ppf(0.975, len(data) - 1))
    half_width = critical * sd / np.sqrt(len(data))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - half_width, mean + half_width]}


def h1_verdict(negative_count: int) -> str:
    return "REPLICATED" if negative_count == 4 else "PARTIAL" if negative_count == 3 else "NOT REPLICATED"


def h2a_verdict(sign_count: int, loo_positive_count: int, loo_mean: float) -> str:
    if sign_count == 4 and loo_positive_count >= 3 and loo_mean > 0:
        return "STRONG"
    if sign_count == 3 or sign_count == 4:
        return "PARTIAL"
    return "NOT SUPPORTED"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_condition(dataset: str, objective: str, seed: int) -> tuple[dict[str, object], dict[str, np.ndarray], Path]:
    root = INPUT_BASE / dataset / objective / f"seed_{seed}"
    manifest_path, arrays_path = root / "manifest.json", root / "per_sample_arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"missing completed condition artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or manifest["backbone_objective"] != objective or manifest["backbone_seed"] != seed:
        raise ValueError(f"manifest provenance mismatch: {root}")
    if manifest["seed_role"] != "confirmatory":
        raise ValueError(f"non-confirmatory condition in analysis: {root}")
    required = {
        "sample_ids", "labels", "folds", "endpoint_adj_margin", "generic_centroid_margin",
        "a_logits", "c_logits", "a_probabilities", "c_probabilities", "a_l1", "c_l1",
        "a_predictive_mean", "c_predictive_mean", "a_inward_shrinkage", "c_inward_shrinkage",
    }
    with np.load(arrays_path) as archive:
        if not required.issubset(set(archive.files)):
            raise ValueError(f"missing per-sample fields: {required - set(archive.files)}")
        arrays = {key: archive[key] for key in required}
    expected_n, expected_end_n = (1080, 66) if dataset == "retina" else (28006, 921)
    if arrays["labels"].shape != (expected_n,) or int((arrays["labels"] == CLASS4).sum()) != expected_end_n:
        raise ValueError(f"unexpected {dataset} class-4 population: {root}")
    if np.unique(arrays["sample_ids"]).size != expected_n:
        raise ValueError(f"sample ID uniqueness failure: {root}")
    if dataset == "retina":
        if not np.array_equal(np.sort(arrays["sample_ids"]), np.arange(1080)):
            raise ValueError(f"sample ID coverage failure: {root}")
        if set(arrays["folds"].tolist()) != set(range(5)):
            raise ValueError(f"OOF fold coverage failure: {root}")
    if not all(np.isfinite(values).all() for values in arrays.values() if np.issubdtype(values.dtype, np.number)):
        raise ValueError(f"non-finite per-sample values: {root}")
    return manifest, arrays, root


def analyze_objective(dataset: str, objective: str, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    h1_rows: list[dict[str, object]] = []
    h2_rows: list[dict[str, object]] = []
    h2b_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    sources: list[str] = []
    for seed in SEEDS:
        manifest, arrays, root = load_condition(dataset, objective, seed)
        sources.append(str(root))
        expected_end_n = 66 if dataset == "retina" else 921
        h1 = h1_delta(arrays["a_l1"], arrays["c_l1"], arrays["labels"], expected_end_n)
        mask = arrays["labels"] == CLASS4
        h1.update({
            "objective": objective,
            "seed": seed,
            "a_mean_inward_shrinkage": float(arrays["a_inward_shrinkage"][mask].mean()),
            "c_mean_inward_shrinkage": float(arrays["c_inward_shrinkage"][mask].mean()),
        })
        h1_rows.append(h1)

        y_c, y_a, generic, ordinal = h2a_vectors(arrays)
        za, _, _ = sample_standardize(y_a)
        zg, _, _ = sample_standardize(generic)
        zo, _, _ = sample_standardize(ordinal)
        _, _, _ = ordinary_least_squares(model_matrix(za, zg), y_c)
        coefficients, _, standard_errors = ordinary_least_squares(model_matrix(za, zg, zo), y_c)
        beta, se = float(coefficients[-1]), float(standard_errors[-1])
        ci = [beta - 1.959963984540054 * se, beta + 1.959963984540054 * se]
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(beta / se))) if se > 0 else 0.0
        mse_m0 = loocv_mse(y_c, y_a, generic, None)
        mse_m1 = loocv_mse(y_c, y_a, generic, ordinal)
        h2_rows.append({
            "objective": objective, "seed": seed, "beta_ord": beta, "hc3_se": se,
            "hc3_ci95_lower": ci[0], "hc3_ci95_upper": ci[1], "nominal_two_sided_p": p_value,
            "mse_loo_m0": mse_m0, "mse_loo_m1": mse_m1, "delta_loo": mse_m0 - mse_m1,
        })
        delta_mu = arrays["c_predictive_mean"][mask] - arrays["a_predictive_mean"][mask]
        delta_p = arrays["c_probabilities"][mask, CLASS4] - arrays["a_probabilities"][mask, CLASS4]
        h2b_rows.append({
            "objective": objective, "seed": seed, "n_class4": int(mask.sum()),
            "mean_delta_mu": float(delta_mu.mean()), "mean_delta_p_end": float(delta_p.mean()),
            "pearson_m_adj_delta_mu": pearson(ordinal, delta_mu),
            "pearson_m_adj_delta_p_end": pearson(ordinal, delta_p),
            "role": "SECONDARY; not an independent H2a test",
        })
        diagnostic_rows.extend({"objective": objective, "seed": seed, "diagnostic": name, "pearson_r": pearson(ordinal, value)} for name, value in {
            "m_adj_vs_p_end_a": arrays["a_probabilities"][mask, CLASS4],
            "m_adj_vs_a_logit_margin_z4_minus_z3": arrays["a_logits"][mask, 4] - arrays["a_logits"][mask, 3],
            "m_adj_vs_generic_g": generic,
            "m_adj_vs_y_a": y_a,
        }.items())

    deltas = [float(row["delta_mae_c_minus_a"]) for row in h1_rows]
    h1_summary = {"objective": objective, "confirmatory_seeds": list(SEEDS), "negative_seed_count": sum(delta < 0 for delta in deltas), **t_interval(deltas)}
    h1_summary["verdict"] = h1_verdict(int(h1_summary["negative_seed_count"]))
    betas = [float(row["beta_ord"]) for row in h2_rows]
    loo = [float(row["delta_loo"]) for row in h2_rows]
    h2_summary = {
        "objective": objective, "confirmatory_seeds": list(SEEDS), "beta_ord_negative_count": sum(beta < 0 for beta in betas),
        "delta_loo_positive_count": sum(value > 0 for value in loo), "mean_delta_loo": float(np.mean(loo)),
        "beta_ord_cross_seed": t_interval(betas),
    }
    h2_summary["verdict"] = h2a_verdict(int(h2_summary["beta_ord_negative_count"]), int(h2_summary["delta_loo_positive_count"]), float(h2_summary["mean_delta_loo"]))
    for filename, payload in (("h1_summary.json", h1_summary), ("h2a_summary.json", h2_summary)):
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "h1_per_seed.csv", h1_rows)
    write_csv(output / "h2a_per_seed.csv", h2_rows)
    write_csv(output / "h2b_secondary.csv", h2b_rows)
    write_csv(output / "diagnostics.csv", diagnostic_rows)
    manifest = {
        "specification": SPECIFICATION, "dataset": dataset, "objective": objective,
        "confirmatory_seeds": list(SEEDS), "replication_unit": "independently trained backbone seed",
        "input_conditions": sources, "h1_primary_effect": "C class-4 L1 MAE minus A class-4 L1 MAE",
        "h2a": "HC3 OLS on true class-4 rows; sample-ddof=1 standardization within seed; LOOCV standardization fitted per training fold",
        "no_training_or_source_artifact_modification": True,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"h1_rows": h1_rows, "h1_summary": h1_summary, "h2_rows": h2_rows, "h2_summary": h2_summary, "h2b_rows": h2b_rows, "diagnostic_rows": diagnostic_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("retina", "solar"), default="retina")
    parser.add_argument("--objective", choices=("ce", "rps", "both"), default="both")
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = args.output_root or OUTPUT_BASE / args.dataset
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite completed analysis output: {output_root}")
    objectives = ("ce", "rps") if args.objective == "both" else (args.objective,)
    results = {objective: analyze_objective(args.dataset, objective, output_root / objective) for objective in objectives}
    combined = {
        "specification": SPECIFICATION, "dataset": args.dataset, "objectives": {
            objective: {"h1_summary": result["h1_summary"], "h2a_summary": result["h2_summary"]}
            for objective, result in results.items()
        },
    }
    (output_root / f"{args.dataset}_confirmatory_summary.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
