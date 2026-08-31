#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,cohen_kappa_score,roc_auc_score
from ordinal_uncertainty.metrics.decision import bayes_decisions
def write(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 root=Path('outputs/retinamnist/native28');out=root/'phase1_75_decision_rule_audit_rerun';out.mkdir(exist_ok=False); all=[]
 for seed in range(5):
  r=list(csv.DictReader((root/'single_model_baseline'/f'seed_{seed}'/'predictions.csv').open()));y=np.array([int(x['true_label']) for x in r]);p=np.array([json.loads(x['probabilities']) for x in r]);d=bayes_decisions(p);rows=[];align=[];cover=[]
  for rule,key,risk in [('mode','mode_decision','mode_l1_risk'),('l1','l1_bayes_decision','l1_bayes_risk'),('l2','l2_bayes_decision','l2_bayes_risk')]:
   pred=d[key];err=np.abs(y-pred); severe=err>=2; row={'seed':seed,'rule':rule,'accuracy':float((y==pred).mean()),'mae':float(err.mean()),'qwk':float(cohen_kappa_score(y,pred,weights='quadratic')),'severe_count':int(severe.sum()),'severe_prevalence':float(severe.mean()),'changed_pct':float((pred!=d['mode_decision']).mean()),'mean_shift':float(np.abs(pred-d['mode_decision']).mean()),'lower_count':int((pred<d['mode_decision']).sum()),'higher_count':int((pred>d['mode_decision']).sum())}; rows.append(row);all.append(row)
   score=d[risk];align.append({'rule':rule,'spearman':float(spearmanr(score,err).statistic),'severe_auroc':float(roc_auc_score(severe,score)),'severe_auprc':float(average_precision_score(severe,score))})
   order=np.argsort(score);cover += [{'rule':rule,'coverage':c,'mae':float(err[order[:max(1,int(np.ceil(c*len(y))))]].mean())} for c in np.arange(1,.09,-.05)]
  sd=out/f'seed_{seed}';sd.mkdir();write(sd/'decision_metrics.csv',rows);write(sd/'risk_alignment.csv',align);write(sd/'risk_coverage.csv',cover)
  write(sd/'predictions_with_decisions.csv',[{'sample_id':i,'true_label':int(y[i]),**{k:(int(v[i]) if 'decision' in k else float(v[i])) for k,v in d.items()}} for i in range(len(y))])
 summary=out/'summary';summary.mkdir();write(summary/'decision_rule_comparison.csv',all);(summary/'phase1_75_summary.json').write_text(json.dumps({'rules':['mode','l1','l2'],'tie_breaking':'smallest minimizer'},indent=2))
if __name__=='__main__':main()
