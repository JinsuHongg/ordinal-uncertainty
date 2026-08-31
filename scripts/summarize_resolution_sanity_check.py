#!/usr/bin/env python3
"""Compare frozen 64px and controlled native-28px seed-0 predictions."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from ordinal_uncertainty.metrics.uncertainty import phase1_5_uncertainty_metrics
def load(path):
    r=list(csv.DictReader(path.open())); y=np.array([int(x['true_label']) for x in r]); pred=np.array([int(x['predicted_label']) for x in r]); p=np.array([json.loads(x['probabilities']) for x in r]); return y,pred,p
def main():
    root=Path('outputs/retinamnist/resolution_sanity_check/seed_0'); paths={'28':root/'size_28','64':Path('outputs/retinamnist/single_model_baseline/seed_0')}; out=root/'comparison'; out.mkdir(exist_ok=False)
    allm={}; rows=[]
    for size,path in paths.items():
        y,pred,p=load(path/'predictions.csv'); e=np.abs(y-pred); u=phase1_5_uncertainty_metrics(p); basic=json.load((path/'metrics.json').open())['prediction_metrics']; basic|={'severe_error_count':int((e>=2).sum()),'severe_error_prevalence':float((e>=2).mean())}; allm[size]=basic
        for name,score in u.items():
            if name=='ordinal_predictive_mean': continue
            order=np.argsort(score); risks=[e[order[:max(1, int(np.ceil(c * len(e))))]].mean() for c in np.arange(1, .09, -.05)]
            rows.append({'size':size,'measure':name,'spearman':float(spearmanr(score,e).statistic),'severe_auroc':float(roc_auc_score(e>=2,score)),'severe_auprc':float(average_precision_score(e>=2,score)),'mae_risk_mean':float(np.mean(risks))})
    (out/'resolution_comparison.json').write_text(json.dumps({'prediction_metrics':allm,'uncertainty':rows},indent=2)+'\n')
    with (out/'resolution_comparison.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__': main()
