#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score,roc_auc_score,average_precision_score
from ordinal_uncertainty.metrics.decision import bayes_decisions
def w(p,r):
 with p.open('w',newline='')as f:x=csv.DictWriter(f,fieldnames=list(r[0]));x.writeheader();x.writerows(r)
def main():
 root=Path('outputs/retinamnist/native28/phase2_model_comparison');sources={'ce':Path('outputs/retinamnist/native28/single_model_baseline/seed_0/predictions.csv'),'coral':root/'coral/seed_0_artifact_complete/evaluation/predictions.csv','rps':root/'rps/seed_0_artifact_complete/evaluation/predictions.csv'};out=root/'seed0_decision_evaluation';out.mkdir(exist_ok=False);all=[];align=[];detect=[];corr=[];coverage=[]
 for name,path in sources.items():
  rows=list(csv.DictReader(path.open()));y=np.array([int(r['true_label'])for r in rows]);p=np.array([json.loads(r['probabilities'])for r in rows]);d=bayes_decisions(p)
  for rule,key,risk in [('mode','mode_decision','mode_l1_risk'),('l1','l1_bayes_decision','l1_bayes_risk'),('l2','l2_bayes_decision','l2_bayes_risk')]:
   pred=d[key];e=np.abs(y-pred);sev=e>=2;s=d[risk];all.append({'model':name,'decision':rule,'accuracy':(y==pred).mean(),'mae':e.mean(),'qwk':cohen_kappa_score(y,pred,weights='quadratic'),'severe_count':sev.sum(),'severe_prevalence':sev.mean()});align.append({'model':name,'decision':rule,'spearman':spearmanr(s,e).statistic});detect.append({'model':name,'decision':rule,'auroc':roc_auc_score(sev,s),'auprc':average_precision_score(sev,s),'severe_count':sev.sum(),'prevalence':sev.mean()});order=np.argsort(s);coverage += [{'model':name,'decision':rule,'coverage':c,'mae':e[order[:max(1,int(np.ceil(c*len(e))))]].mean()}for c in np.arange(1,.09,-.05)]
  m,l=d['mode_decision'],d['l1_bayes_decision'];shift=l-m;corr.append({'model':name,'changed_fraction':(shift!=0).mean(),'mean_abs_shift':np.abs(shift).mean(),'shift_1':(np.abs(shift)==1).mean(),'shift_2plus':(np.abs(shift)>=2).mean(),'up':(shift>0).mean(),'down':(shift<0).mean()})
 w(out/'decision_metrics.csv',all);w(out/'risk_alignment.csv',align);w(out/'severe_detection.csv',detect);w(out/'decision_correction.csv',corr);w(out/'risk_coverage.csv',coverage);(out/'seed0_summary.json').write_text(json.dumps({'sources':{k:str(v)for k,v in sources.items()}},indent=2))
if __name__=='__main__':main()
