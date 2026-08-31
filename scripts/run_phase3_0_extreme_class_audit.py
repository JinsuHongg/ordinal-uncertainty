#!/usr/bin/env python3
"""Frozen-prediction Phase 3.0 extreme-class diagnostic audit."""
from __future__ import annotations

import csv
import json
import sys
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import class_probability_means, inward_shrinkage, ordinal_bias

SEEDS = range(5)
K = 5


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)


def read_state(model: str, seed: int, calibration: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path("outputs/retinamnist/native28/phase2_5_calibration_audit") / model / f"seed_{seed}" / "predictions_calibrated.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labels = np.asarray([int(row["true_label"]) for row in rows])
    identifiers = np.asarray([int(row["sample_id"]) for row in rows])
    column = "raw_probabilities" if calibration == "raw" else "temperature_scaled_probabilities"
    probabilities = np.asarray([json.loads(row[column]) for row in rows], dtype=float)
    if not np.allclose(probabilities.sum(1), 1, atol=1e-6) or np.any(probabilities < 0):
        raise ValueError(f"invalid {model}/{seed}/{calibration} probabilities")
    return identifiers, labels, probabilities


def stats(values: np.ndarray) -> dict[str, float | None]:
    return {"mean": float(values.mean()) if len(values) else None, "median": float(np.median(values)) if len(values) else None, "std": float(values.std(ddof=1)) if len(values) > 1 else None, "q25": float(np.quantile(values, .25)) if len(values) else None, "q75": float(np.quantile(values, .75)) if len(values) else None}


def aggregate(rows: list[dict], group_keys: list[str], value_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows: groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    result = []
    for key, members in groups.items():
        item = dict(zip(group_keys, key))
        for value in value_keys:
            values = np.asarray([member[value] for member in members if member.get(value) is not None], dtype=float)
            item[f"{value}_mean"] = float(values.mean()) if len(values) else None
            item[f"{value}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/retinamnist/native28/phase3_0_extreme_class_audit")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    loaders, metadata = retinamnist_loaders(Path("data/medmnist"), 128, 0, False, 28)
    split_rows = []
    for split, dataset in ((key, loader.dataset) for key, loader in loaders.items()):
        counts = Counter(dataset.labels.reshape(-1).tolist()); largest = max(counts.values())
        for true_class in range(K): split_rows.append({"split": split, "true_class": true_class, "count": int(counts[true_class]), "percentage": counts[true_class] / len(dataset), "ratio_to_largest": counts[true_class] / largest})
    reference: tuple[np.ndarray, np.ndarray] | None = None
    class_metrics=[]; confusion=[]; error_distance=[]; mean_probability=[]; probability_mass=[]; geometry=[]; biases=[]; risks=[]; within_class=[]
    for seed in SEEDS:
        seed_output = output / f"seed_{seed}"; seed_output.mkdir()
        for model in ("ce", "rps"):
            for calibration in ("raw", "temperature_scaled"):
                identifiers, labels, probabilities = read_state(model, seed, calibration)
                if reference is None: reference = identifiers, labels
                elif not (np.array_equal(reference[0], identifiers) and np.array_equal(reference[1], labels)): raise ValueError("frozen prediction artifacts are not aligned")
                decisions = bayes_decisions(probabilities); predictive_mean = probabilities @ np.arange(K); shrinkage = inward_shrinkage(labels, probabilities); class_bias = ordinal_bias(labels, probabilities)
                for true_class, vector in enumerate(class_probability_means(labels, probabilities)):
                    mean_probability.append({"seed":seed,"model":model,"calibration":calibration,"true_class":true_class,**{f"p{predicted_class}":float(vector[predicted_class]) for predicted_class in range(K)}})
                for decision_name, key in (("mode","mode_decision"),("l1","l1_bayes_decision"),("l2","l2_bayes_decision")):
                    prediction=decisions[key]; error=np.abs(labels-prediction); decision_bias=prediction-labels
                    for true_class in range(K):
                        mask=labels==true_class; values=error[mask]; severe=values>=2
                        class_metrics.append({"seed":seed,"model":model,"calibration":calibration,"decision":decision_name,"true_class":true_class,"count":int(mask.sum()),"accuracy":float((prediction[mask]==labels[mask]).mean()),"mae":float(values.mean()),"severe_count":int(severe.sum()),"severe_prevalence":float(severe.mean())})
                        biases.append({"seed":seed,"model":model,"calibration":calibration,"decision":decision_name,"true_class":true_class,"predictive_mean_bias":float(class_bias[true_class]),"decision_bias":float(decision_bias[mask].mean()),"inward_shrinkage":float(shrinkage[mask].mean())})
                        for distance in range(K): error_distance.append({"seed":seed,"model":model,"calibration":calibration,"decision":decision_name,"true_class":true_class,"ordinal_error":distance,"count":int((values==distance).sum()),"fraction":float((values==distance).mean())})
                        for predicted_class in range(K):
                            count=int(((labels==true_class)&(prediction==predicted_class)).sum())
                            confusion.append({"seed":seed,"model":model,"calibration":calibration,"decision":decision_name,"true_class":true_class,"predicted_class":predicted_class,"count":count,"row_fraction":count/int(mask.sum())})
                l1_prediction=decisions["l1_bayes_decision"]; l1_error=np.abs(labels-l1_prediction); l1_risk=decisions["l1_bayes_risk"]
                for true_class in range(K):
                    mask=labels==true_class; p_true=probabilities[mask,true_class]; risk=l1_risk[mask]
                    entry={"seed":seed,"model":model,"calibration":calibration,"true_class":true_class,"count":int(mask.sum()),**{f"true_probability_{key}":value for key,value in stats(p_true).items()},**{f"l1_risk_{key}":value for key,value in stats(risk).items()},"mean_predictive_mean":float(predictive_mean[mask].mean()),"median_predictive_mean":float(np.median(predictive_mean[mask])),"mean_inward_shrinkage":float(shrinkage[mask].mean())}
                    if true_class == 4: entry |= {"mean_p3_plus_p4":float(probabilities[mask,3:].sum(1).mean()),"mean_p2_plus_p3_plus_p4":float(probabilities[mask,2:].sum(1).mean())}
                    if true_class == 0: entry |= {"mean_p0_plus_p1":float(probabilities[mask,:2].sum(1).mean()),"mean_p0_plus_p1_plus_p2":float(probabilities[mask,:3].sum(1).mean())}
                    probability_mass.append(entry)
                    target=(l1_error[mask]>=2)
                    within_class.append({"seed":seed,"model":model,"calibration":calibration,"true_class":true_class,"count":int(mask.sum()),"l1_risk_error_spearman":float(spearmanr(risk,l1_error[mask]).statistic) if len(np.unique(l1_error[mask]))>1 else None,"severe_count":int(target.sum()),"severe_auroc":float(roc_auc_score(target,risk)) if len(np.unique(target))==2 else None,"severe_auprc":float(average_precision_score(target,risk)) if len(np.unique(target))==2 else None})
                    for rule,key in (("mode","mode_decision"),("l1","l1_bayes_decision"),("l2","l2_bayes_decision")):
                        for action in range(K): geometry.append({"seed":seed,"model":model,"calibration":calibration,"true_class":true_class,"quantity":rule,"value":action,"count":int(((labels==true_class)&(decisions[key]==action)).sum()),"fraction":float(((labels==true_class)&(decisions[key]==action)).sum()/mask.sum())})
        # compact per-seed references without duplicating historical predictions
        (seed_output / "sources.json").write_text(json.dumps({"seed":seed,"models":["ce","rps"],"states":["raw","temperature_scaled"],"test_count":int(len(reference[1]))},indent=2)+"\n")
    summary=output/"summary"; summary.mkdir()
    for name,rows in (("split_class_counts.csv",split_rows),("classwise_metrics.csv",class_metrics),("confusion_matrices.csv",confusion),("error_distance_by_class.csv",error_distance),("mean_probability_by_true_class.csv",mean_probability),("extreme_probability_mass.csv",probability_mass),("ordinal_bias.csv",biases),("decision_geometry.csv",geometry),("extreme_risk_summary.csv",within_class)):
        write_csv(summary/name,rows)
    key_class4=[row for row in probability_mass if row["true_class"] in (0,4)]
    write_csv(summary/"extreme_class_comparison.csv",aggregate(key_class4,["model","calibration","true_class"],["true_probability_mean","mean_predictive_mean","mean_inward_shrinkage","l1_risk_mean","mean_p3_plus_p4","mean_p2_plus_p3_plus_p4","mean_p0_plus_p1","mean_p0_plus_p1_plus_p2"]))
    write_csv(summary/"classwise_metrics_summary.csv",aggregate(class_metrics,["model","calibration","decision","true_class"],["accuracy","mae","severe_prevalence"]))
    write_csv(summary/"bias_summary.csv",aggregate(biases,["model","calibration","decision","true_class"],["predictive_mean_bias","decision_bias","inward_shrinkage"]))
    (summary/"phase3_0_summary.json").write_text(json.dumps({"frozen_models":["ce","rps"],"calibration_states":["raw","temperature_scaled"],"seeds":list(SEEDS),"test_count":int(len(reference[1])),"split_metadata":metadata,"training_performed":False},indent=2)+"\n")


if __name__ == "__main__": main()
