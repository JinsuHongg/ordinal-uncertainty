#!/usr/bin/env python3
"""Evaluate Phase 1.5 measures from frozen Experiment 0 prediction CSVs."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from ordinal_uncertainty.metrics.uncertainty import phase1_5_uncertainty_metrics

FAMILIES = {"predictive_entropy":"nominal", "confidence_uncertainty":"nominal", "margin_uncertainty":"nominal", "ordinal_variance":"simple_ordinal", "ordinal_absolute_deviation":"simple_ordinal", "prediction_distance_l1":"exploratory_simple_ordinal", "ocs_entropy":"literature_ordinal", "ocs_variance":"literature_ordinal", "consensus_cns_dissention":"literature_ordinal", "consensus_c2_dispersion":"literature_ordinal", "bayes_risk_l2":"literature_ordinal"}
def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment-root',default='outputs/retinamnist/single_model_baseline'); ap.add_argument('--output-root',default='outputs/retinamnist/phase1_5_ordinal_uq_audit'); args=ap.parse_args()
    source, root=Path(args.experiment_root),Path(args.output_root)
    if root.exists(): raise FileExistsError(f'Refusing to overwrite {root}')
    all_rows=[]
    for seed in range(5):
        records=list(csv.DictReader((source/f'seed_{seed}'/'predictions.csv').open()))
        y=np.array([int(r['true_label']) for r in records]); pred=np.array([int(r['predicted_label']) for r in records]); p=np.array([json.loads(r['probabilities']) for r in records]); error=np.abs(y-pred); u=phase1_5_uncertainty_metrics(p)
        out=root/f'seed_{seed}'; out.mkdir(parents=True)
        metric_rows=[]; severity=[]; curves=[]; classwise=[]
        for name, score in u.items():
            if name == 'ordinal_predictive_mean': continue
            row={'seed':seed,'measure':name,'family':FAMILIES[name],'spearman':float(spearmanr(score,error).statistic)}
            for target_name,target in [('any_error',error>0),('severe_error',error>=2)]:
                row[f'{target_name}_count']=int(target.sum()); row[f'{target_name}_prevalence']=float(target.mean()); row[f'{target_name}_auroc']=float(roc_auc_score(target,score)); row[f'{target_name}_auprc']=float(average_precision_score(target,score))
            metric_rows.append(row); all_rows.append(row)
            for label,mask in [('correct',error==0),('adjacent',error==1),('severe',error>=2)]: severity.append({'measure':name,'family':FAMILIES[name],'error_severity':label,'count':int(mask.sum()),'mean':float(score[mask].mean()),'std':float(score[mask].std(ddof=1)) if mask.sum()>1 else None,'median':float(np.median(score[mask]))})
            order=np.argsort(score,kind='stable')
            for coverage in np.arange(1,.09,-.05):
                keep=order[:max(1,int(np.ceil(coverage*len(y))))]; curves.append({'measure':name,'family':FAMILIES[name],'coverage':round(float(coverage),2),'retained_count':len(keep),'classification_risk':float((y[keep]!=pred[keep]).mean()),'ordinal_risk_mae':float(error[keep].mean())})
        for cls in range(p.shape[1]):
            mask=y==cls
            for name,score in u.items():
                if name != 'ordinal_predictive_mean': classwise.append({'true_class':cls,'measure':name,'family':FAMILIES[name],'count':int(mask.sum()),'mean_uncertainty':float(score[mask].mean()),'severe_error_count':int((error[mask]>=2).sum()),'severe_error_rate':float((error[mask]>=2).mean())})
        write(out/'uncertainty_metrics.csv',metric_rows); write(out/'detection_metrics.csv',metric_rows); write(out/'uncertainty_by_error.csv',severity); write(out/'risk_coverage.csv',curves); write(out/'classwise_metrics.csv',classwise)
    summary=root/'summary'; summary.mkdir(); write(summary/'baseline_comparison.csv',all_rows)
    ranks=[]
    for measure in FAMILIES:
        rows=[r for r in all_rows if r['measure']==measure]
        result={'measure':measure,'family':FAMILIES[measure]}
        for key in ['spearman','any_error_auroc','any_error_auprc','severe_error_auroc','severe_error_auprc']:
            v=np.array([r[key] for r in rows]); result[key+'_mean']=float(v.mean()); result[key+'_std']=float(v.std(ddof=1))
        # Mean MAE risk over a fixed grid; lower is better.
        risks=[]
        for seed in range(5): risks.extend([float(r['ordinal_risk_mae']) for r in csv.DictReader((root/f'seed_{seed}'/'risk_coverage.csv').open()) if r['measure']==measure])
        result['ordinal_mae_risk_auc_mean']=float(np.mean(risks)); ranks.append(result)
    ranks.sort(key=lambda r:r['severe_error_auprc_mean'],reverse=True); write(summary/'ranking_summary.csv',ranks)
    (summary/'phase1_5_summary.json').write_text(json.dumps({'source_experiment':str(source),'seeds':[0,1,2,3,4],'measures':FAMILIES,'primary_ranking':'severe_error_auprc_mean'},indent=2)+'\n')
if __name__=='__main__': main()
