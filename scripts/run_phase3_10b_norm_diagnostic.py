#!/usr/bin/env python3
"""Declared one-shot norm diagnostic for the completed Phase 3.10B audit."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from ordinal_uncertainty.evaluation.head_audit import common_norm_weights, linear_logits
from ordinal_uncertainty.metrics.decision import bayes_decisions

P10A=Path("outputs/retinamnist/phase3_10a_rop_objective_falsification")
OUT=Path("outputs/retinamnist/phase3_10b_head_bias_localization_audit/counterfactual_swaps/norm_diagnostic.csv")

def main() -> None:
    if OUT.exists(): raise FileExistsError("refusing to overwrite norm diagnostic")
    with np.load(P10A/"frozen_features/train_rps_features.npz") as x: ids, labels, features=x['sample_id'],x['labels'],x['features']
    folds=np.asarray([int(r['fold']) for r in csv.DictReader((P10A/'fold_assignments/assignments.csv').open())])
    logits=np.empty((len(labels),5)); rows=[]
    for fold in range(5):
        state=torch.load(P10A/'condition_b_balanced_head/fold_checkpoints'/f'fold_{fold}.pt',map_location='cpu',weights_only=False)['head_state_dict']
        w,b=state['weight'].numpy(),state['bias'].numpy(); reference=float(np.linalg.norm(w,axis=1).mean())
        _, logits[folds==fold]=linear_logits(features[folds==fold],common_norm_weights(w,reference),b)
        rows.append({'fold':fold,'balanced_mean_norm':reference,'normalized_common_norm':reference})
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True); d=bayes_decisions(p)['l1_bayes_decision'];err=np.abs(labels-d)
    c4,c0=labels==4,labels==0
    rows.append({'fold':'pooled','global_mae_l1':float(err.mean()),'qwk_l1':float(cohen_kappa_score(labels,d,weights='quadratic')),'class4_mae_l1':float(err[c4].mean()),'class4_exact_l1':int((d[c4]==4).sum()),'class4_routing_l1':json.dumps([int((d[c4]==k).sum()) for k in range(5)]),'class0_mae_l1':float(err[c0].mean())})
    with OUT.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
if __name__=='__main__': main()
