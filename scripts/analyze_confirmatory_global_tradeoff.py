#!/usr/bin/env python3
"""Deterministic global trade-off analysis from frozen confirmatory A/C arrays."""
from __future__ import annotations

import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

INPUT = Path("outputs/mechanism_replication/ac")
OUTPUT = Path("outputs/mechanism_replication/analysis/global_tradeoff")
SETS = (("retina", "ce", 1080, 66), ("retina", "rps", 1080, 66),
        ("solar", "ce", 28006, 921), ("solar", "rps", 28006, 921))
SEEDS = (1,2,3,4); K=5; END=4

def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)

def metrics(y, d, p):
    err=np.abs(y-d); out={"mae":float(err.mean()),"severe_rate":float((err>=2).mean())}
    out["macro_mae"]=float(np.mean([err[y==k].mean() for k in range(K)]))
    for k in range(K):
        mask=y==k; out[f"mae_{k}"]=float(err[mask].mean()); out[f"recall_{k}"]=float((d[mask]==k).mean()); out[f"severe_{k}"]=float((err[mask]>=2).mean()); out[f"p_end_{k}"]=float(p[mask,END].mean()); out[f"true_p_{k}"]=float(p[mask,k].mean())
    return out

def main():
    if OUTPUT.exists(): raise FileExistsError(f"refusing overwrite: {OUTPUT}")
    allrows=[]; sources=[]
    for ds,obj,n,nend in SETS:
        setting=[]; perclass=[]; routing=[]; mass=[]
        for seed in SEEDS:
            root=INPUT/ds/obj/f"seed_{seed}"; mp=root/"manifest.json"; ap=root/"per_sample_arrays.npz"
            manifest=json.loads(mp.read_text()); assert manifest["dataset"]==ds and manifest["backbone_objective"]==obj and manifest["backbone_seed"]==seed and manifest["seed_role"]=="confirmatory"
            z=np.load(ap); need={"sample_ids","labels","a_l1","c_l1","a_probabilities","c_probabilities"}; assert need<=set(z.files)
            y=z["labels"]; a=z["a_l1"]; c=z["c_l1"]; pa=z["a_probabilities"]; pc=z["c_probabilities"]
            assert len(y)==n and len(np.unique(z["sample_ids"]))==n and (y==END).sum()==nend and np.all(np.isfinite(pa)) and np.all(np.isfinite(pc))
            A=metrics(y,a,pa); C=metrics(y,c,pc); r={"dataset":ds,"objective":obj,"seed":seed,"endpoint_mae_A":A["mae_4"],"endpoint_mae_C":C["mae_4"],"delta_endpoint_mae":C["mae_4"]-A["mae_4"],"global_mae_A":A["mae"],"global_mae_C":C["mae"],"delta_global_mae":C["mae"]-A["mae"],"macro_mae_A":A["macro_mae"],"macro_mae_C":C["macro_mae"],"delta_macro_mae":C["macro_mae"]-A["macro_mae"],"severe_A":A["severe_rate"],"severe_C":C["severe_rate"],"delta_severe":C["severe_rate"]-A["severe_rate"]}; setting.append(r); allrows.append(r)
            for k in range(K):
                q={"dataset":ds,"objective":obj,"seed":seed,"true_class":k,**{f"A_{x}":A[f"{x}_{k}"] for x in ("mae","recall","severe","p_end","true_p")},**{f"C_{x}":C[f"{x}_{k}"] for x in ("mae","recall","severe","p_end","true_p")}}
                for x in ("mae","recall","severe","p_end","true_p"): q[f"delta_{x}"]=q[f"C_{x}"]-q[f"A_{x}"]
                perclass.append(q); mask=y==k
                for pred in range(K): routing.append({"dataset":ds,"objective":obj,"seed":seed,"true_class":k,"predicted_class":pred,"A_count":int((a[mask]==pred).sum()),"C_count":int((c[mask]==pred).sum())})
            sources.append({"path":str(ap),"sha256":hashlib.sha256(ap.read_bytes()).hexdigest()})
        rows(OUTPUT/ds/obj/"seed_level_summary.csv",setting); rows(OUTPUT/ds/obj/"per_class_metrics.csv",perclass); rows(OUTPUT/ds/obj/"routing_summary.csv",routing); rows(OUTPUT/ds/obj/"endpoint_mass_shift.csv",[{k:v for k,v in x.items() if k in ("dataset","objective","seed","true_class","A_p_end","C_p_end","delta_p_end")} for x in perclass])
    rows(OUTPUT/"summary/global_tradeoff_all_settings.csv",allrows)
    (OUTPUT/"analysis_manifest.json").write_text(json.dumps({"sources":sources,"decision_rule":"saved exact discrete L1 decisions","evaluation_populations":{"retina":"1080 training-only OOF rows; 66 endpoint","solar":"28006 fixed archived-readout rows; 921 endpoint"},"class_count":5,"endpoint_class":4,"no_retraining_or_head_fitting":True,"script":"scripts/analyze_confirmatory_global_tradeoff.py","timestamp_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
if __name__=="__main__": main()
