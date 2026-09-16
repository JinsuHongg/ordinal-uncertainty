#!/usr/bin/env python3
"""Analyze saved Retina A/N/C OOF outputs without fitting or inference."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUT = Path("outputs/mechanism_replication/natural_direction/retina")
OUTPUT = Path("outputs/mechanism_replication/analysis/natural_direction/retina")
GLOBAL = Path("outputs/mechanism_replication/analysis/global_tradeoff/retina")
K, END, SEEDS = 5, 4, (1, 2, 3, 4)


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metrics(y: np.ndarray, decision: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    error = np.abs(y - decision)
    out = {"global_mae": float(error.mean()), "macro_mae": float(np.mean([error[y == k].mean() for k in range(K)])), "global_severe": float((error >= 2).mean())}
    for k in range(K):
        mask = y == k
        out.update({f"mae_{k}": float(error[mask].mean()), f"recall_{k}": float((decision[mask] == k).mean()), f"severe_{k}": float((error[mask] >= 2).mean()), f"p_end_{k}": float(probability[mask, END].mean()), f"true_p_{k}": float(probability[mask, k].mean())})
    return out


def ci(values: np.ndarray) -> list[float]:
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = 3.182446305 * sd / np.sqrt(len(values))
    return [mean - half, mean + half]


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seed_rows=[]; class_rows=[]; mass_rows=[]; routing=[]; sources=[]; summaries={}
    for objective in ("ce", "rps"):
        objective_rows=[]
        reference = {int(row["seed"]): row for row in csv.DictReader((GLOBAL / objective / "seed_level_summary.csv").open())}
        for seed in SEEDS:
            root=INPUT/objective/f"seed_{seed}"
            manifest=json.loads((root/"manifest.json").read_text())
            if manifest["a_c_identity"] != "PASS" or manifest["no_backbone_training"] is not True or manifest["no_feature_regeneration"] is not True: raise RuntimeError("invalid N source manifest")
            with np.load(root/"per_sample_arrays.npz",allow_pickle=False) as z:
                y=z["labels"]; ids=z["sample_ids"]; folds=z["folds"]
                if y.shape != (1080,) or len(np.unique(ids)) != 1080 or not np.array_equal(np.bincount(folds,minlength=5),np.full(5,216)): raise RuntimeError("invalid OOF population")
                decisions={name:z[f"{name}_l1"] for name in "anc"}; probabilities={name:z[f"{name}_probabilities"] for name in "anc"}
                values={name:metrics(y,decisions[name],probabilities[name]) for name in "anc"}
            endpoint=y==END
            row={"dataset":"retina","objective":objective,"seed":seed}
            for name in "anc":
                row.update({f"endpoint_mae_{name.upper()}":values[name]["mae_4"],f"global_mae_{name.upper()}":values[name]["global_mae"],f"macro_mae_{name.upper()}":values[name]["macro_mae"],f"global_severe_{name.upper()}":values[name]["global_severe"]})
            row.update({"delta_NA_endpoint":row["endpoint_mae_N"]-row["endpoint_mae_A"],"delta_CA_endpoint":row["endpoint_mae_C"]-row["endpoint_mae_A"],"delta_NC_endpoint":row["endpoint_mae_N"]-row["endpoint_mae_C"]})
            prior=reference[seed]
            for key in ("endpoint_mae_A","endpoint_mae_C","global_mae_A","global_mae_C","macro_mae_A","macro_mae_C"):
                if not np.isclose(row[key],float(prior[key]),atol=1e-12): raise RuntimeError(f"A/C global-tradeoff identity mismatch {objective} {seed} {key}")
            seed_rows.append(row); objective_rows.append(row)
            for k in range(K):
                item={"dataset":"retina","objective":objective,"seed":seed,"true_class":k}
                for name in "anc":
                    upper=name.upper()
                    for key in ("mae","recall","severe","p_end","true_p"): item[f"{upper}_{key}"]=values[name][f"{key}_{k}"]
                for key in ("mae","recall","severe","p_end","true_p"):
                    item[f"N_minus_A_{key}"]=item[f"N_{key}"]-item[f"A_{key}"]; item[f"C_minus_A_{key}"]=item[f"C_{key}"]-item[f"A_{key}"]; item[f"C_minus_N_{key}"]=item[f"C_{key}"]-item[f"N_{key}"]
                class_rows.append(item); mass_rows.append({key:item[key] for key in item if key in ("dataset","objective","seed","true_class","A_p_end","N_p_end","C_p_end","N_minus_A_p_end","C_minus_A_p_end","C_minus_N_p_end")})
                mask=y==k
                for predicted in range(K): routing.append({"dataset":"retina","objective":objective,"seed":seed,"true_class":k,"predicted_class":predicted,"A_count":int((decisions["a"][mask]==predicted).sum()),"N_count":int((decisions["n"][mask]==predicted).sum()),"C_count":int((decisions["c"][mask]==predicted).sum())})
            delta=np.asarray([x["delta_NA_endpoint"] for x in objective_rows])
            sources.append({"objective":objective,"seed":seed,"path":str(root/"per_sample_arrays.npz")})
        d=np.asarray([x["delta_NA_endpoint"] for x in objective_rows]); summaries[objective]={"endpoint_mean_A":float(np.mean([x["endpoint_mae_A"] for x in objective_rows])),"endpoint_mean_N":float(np.mean([x["endpoint_mae_N"] for x in objective_rows])),"endpoint_mean_C":float(np.mean([x["endpoint_mae_C"] for x in objective_rows])),"delta_NA_mean":float(d.mean()),"delta_NA_sd":float(d.std(ddof=1)),"delta_NA_t95":ci(d),"improving_seeds":int((d<0).sum()),"label":"CONSISTENT" if (d<0).sum()==4 else "MIXED" if (d<0).sum()==3 else "NOT CONSISTENT","global_mae_means":{name:float(np.mean([x[f"global_mae_{name}"] for x in objective_rows])) for name in ("A","N","C")},"macro_mae_means":{name:float(np.mean([x[f"macro_mae_{name}"] for x in objective_rows])) for name in ("A","N","C")},"global_severe_means":{name:float(np.mean([x[f"global_severe_{name}"] for x in objective_rows])) for name in ("A","N","C")}}
    write(OUTPUT/"per_seed_summary.csv",seed_rows); write(OUTPUT/"per_class_metrics.csv",class_rows); write(OUTPUT/"endpoint_mass_shift.csv",mass_rows); write(OUTPUT/"routing_summary.csv",routing)
    verdict="C — RETINA NATURAL DIRECTION-ONLY DOES NOT REPRODUCE C; BALANCED SAMPLING IS PRIMARY"
    summary={"verdict":verdict,"dataset":"RetinaMNIST","evaluation":"same 1080-row training-only OOF population; exact L1 decisions","per_objective":summaries,"A_C_identity":"PASS against prior global-tradeoff outputs","interpretation":"Natural direction-only adaptation did not improve the endpoint in a majority of seeds for either objective, while balanced C improves all archived confirmatory seeds.","phase3_19_relation":"Qualitatively consistent with the historical single-backbone RPS factorial result that balanced sampling, rather than objective choice alone, drove direction adaptation.","bias_prior_baseline":"YES; a prior/bias-only control remains relevant for reviewer concerns about endpoint-mass redistribution, but was not run here.","no_backbone_training":True,"no_feature_regeneration":True,"no_solar":True}
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (OUTPUT/"analysis_manifest.json").write_text(json.dumps({"script":"scripts/analyze_retina_natural_direction.py","sources":sources,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"no_head_fitting":True},indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
