#!/usr/bin/env python3
"""Scientific seed-0 comparison from frozen CE/RPS/WCE/SLACE predictions."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics

COVERAGES = np.round(np.arange(1.0, 0.099, -0.05), 2)
ROOT = Path("outputs/retinamnist/native28")
SOURCES = {
    "ce": ROOT / "single_model_baseline/seed_0/predictions.csv",
    "rps": ROOT / "phase2_model_comparison/rps/seed_0_artifact_complete/evaluation/predictions.csv",
    "weighted_ce": ROOT / "phase3_1_existing_baselines/weighted_ce/seed_0/evaluation/predictions.csv",
    "slace": ROOT / "phase3_1_existing_baselines/slace/seed_0/evaluation/predictions.csv",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ids = np.asarray([int(row["sample_id"]) for row in rows])
    labels = np.asarray([int(row["true_label"]) for row in rows])
    probabilities = np.asarray([json.loads(row["probabilities"]) for row in rows], dtype=float)
    if len(rows) != 400 or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise ValueError(f"invalid prediction artifact: {path}")
    # CSV round trips can leave sums a few float32 ulps from one. Normalize only
    # after validating the stored vectors, so downstream sklearn metrics receive
    # an exact simplex representation without masking malformed artifacts.
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return ids, labels, probabilities


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / "phase3_1_existing_baselines/summary"
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    reference = None; predictive_rows=[]; decision_rows=[]; extreme_rows=[]; error_rows=[]; risk_rows=[]
    for method, path in SOURCES.items():
        ids, labels, p = read_predictions(path)
        if reference is None: reference = ids, labels
        elif not (np.array_equal(ids, reference[0]) and np.array_equal(labels, reference[1])): raise ValueError(f"sample mismatch: {method}")
        predictive = prediction_metrics(labels, p); predictive["ece"] = expected_calibration_error(labels, p)[0]
        d = bayes_decisions(p); mode_error=np.abs(labels-d["mode_decision"])
        predictive_rows.append({"method":method,**predictive,"severe_count":int((mode_error>=2).sum()),"severe_prevalence":float((mode_error>=2).mean())})
        mu=p@np.arange(p.shape[1]); shrink=inward_shrinkage(labels,p)
        for rule,key,risk_key in (("mode","mode_decision","mode_l1_risk"),("l1","l1_bayes_decision","l1_bayes_risk"),("l2","l2_bayes_decision","l2_bayes_risk")):
            pred=d[key]; error=np.abs(labels-pred); severe=error>=2; score=d[risk_key]; order=np.argsort(score,kind="stable")
            selective=[float(error[order[:max(1,int(np.ceil(c*len(labels))))]].mean()) for c in COVERAGES]
            decision_rows.append({"method":method,"decision":rule,"accuracy":float((pred==labels).mean()),"mae":float(error.mean()),"qwk":float(cohen_kappa_score(labels,pred,weights="quadratic")),"severe_count":int(severe.sum()),"severe_prevalence":float(severe.mean())})
            risk_rows.append({"method":method,"decision":rule,"spearman":float(spearmanr(score,error).statistic),"severe_auroc":float(roc_auc_score(severe,score)),"severe_auprc":float(average_precision_score(severe,score)),"mean_mae_selective_risk":float(np.mean(selective)),"mode_to_l1_changed_fraction":float((d['mode_decision']!=d['l1_bayes_decision']).mean()) if rule=="l1" else None})
            for true_class in (0,4):
                mask=labels==true_class
                for predicted_class in range(5):
                    error_rows.append({"method":method,"decision":rule,"true_class":true_class,"predicted_class":predicted_class,"count":int(((pred==predicted_class)&mask).sum()),"fraction":float((pred[mask]==predicted_class).mean()),"ordinal_error":abs(true_class-predicted_class)})
        for true_class in (0,4):
            mask=labels==true_class; pred=d['l1_bayes_decision']; error=np.abs(labels[mask]-pred[mask]); risk=d['l1_bayes_risk'][mask]; true_p=p[mask,true_class]; near=p[mask,:2].sum(1) if true_class==0 else p[mask,3:].sum(1); broad=p[mask,:3].sum(1) if true_class==0 else p[mask,2:].sum(1)
            extreme_rows.append({"method":method,"true_class":true_class,"count":int(mask.sum()),"accuracy":float((pred[mask]==true_class).mean()),"mae":float(error.mean()),"severe_count":int((error>=2).sum()),"severe_prevalence":float((error>=2).mean()),"true_probability_mean":float(true_p.mean()),"true_probability_median":float(np.median(true_p)),"true_probability_std":float(true_p.std(ddof=1)),"near_mass_mean":float(near.mean()),"near_mass_median":float(np.median(near)),"near_mass_std":float(near.std(ddof=1)),"broad_mass_mean":float(broad.mean()),"predictive_mean_mean":float(mu[mask].mean()),"predictive_mean_median":float(np.median(mu[mask])),"predictive_mean_std":float(mu[mask].std(ddof=1)),"inward_shrinkage":float(shrink[mask].mean()),"decision_bias":float((pred[mask]-true_class).mean()),"l1_risk_mean":float(risk.mean()),"l1_risk_median":float(np.median(risk)),"l1_risk_std":float(risk.std(ddof=1))})
    write_csv(output/"phase3_1_seed0_comparison.csv",predictive_rows);write_csv(output/"phase3_1_decision_comparison.csv",decision_rows);write_csv(output/"phase3_1_extreme_class_comparison.csv",extreme_rows);write_csv(output/"phase3_1_error_distance.csv",error_rows);write_csv(output/"phase3_1_risk_comparison.csv",risk_rows)
    duplicate_root=ROOT/"phase3_1_existing_baselines/slace"; duplicates=[duplicate_root/name for name in ("seed_0","seed_0_complete","seed_0_diagnostic")]
    evidence={"canonical_slace":str(duplicates[0]),"duplicate_training_history_sha256":[digest(path/"training_history.csv") for path in duplicates],"duplicate_metrics_sha256":[digest(path/"evaluation/metrics.json") for path in duplicates],"samples_aligned":True,"test_count":400,"training_performed":False}
    (output/"phase3_1_summary.json").write_text(json.dumps(evidence,indent=2)+"\n")


if __name__ == "__main__": main()
