#!/usr/bin/env python3
"""Seed-0 validation runner for published WCE and SLACE baselines."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.evaluation.probability_pipeline import finalize_probability_evaluation
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.metrics.slace import inverse_frequency_weights, slace_loss, slace_matrices, weighted_cross_entropy
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.utils.reproducibility import set_seed

def write(path, rows):
 if rows:
  with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('method',choices=['weighted_ce','slace']);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 set_seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); loaders,meta=retinamnist_loaders(Path('data/medmnist'),64,0,False,28); counts=torch.tensor([Counter(loaders['train'].dataset.labels.reshape(-1).tolist())[i] for i in range(meta['num_classes'])]); weights=inverse_frequency_weights(counts); prox,soft,dom=slace_matrices(counts,1.0); model=make_resnet18(meta['num_classes']).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); history=[];best=float('inf');state=None;best_epoch=0
 def loss_fn(z,y): return weighted_cross_entropy(z,y,weights) if a.method=='weighted_ce' else slace_loss(z,y,soft,dom)
 for epoch in range(1,a.epochs+1):
  model.train();tl=[]
  for x,y in loaders['train']:
   y=y.reshape(-1).long().to(dev); z=model(x.to(dev)); loss=loss_fn(z,y);opt.zero_grad();loss.backward();opt.step();tl.append(loss.item())
  model.eval();vl=[]
  with torch.no_grad():
   for x,y in loaders['val']: vl.append(loss_fn(model(x.to(dev)),y.reshape(-1).long().to(dev)).item())
  value=float(np.mean(vl));history.append({'epoch':epoch,'train_loss':float(np.mean(tl)),'validation_loss':value})
  if value<best:best=value;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state);model.eval();zs=[];ys=[]
 with torch.no_grad():
  for x,y in loaders['test']:zs.append(model(x.to(dev)).cpu());ys.append(y.reshape(-1))
 logits=torch.cat(zs);labels=torch.cat(ys).numpy();finalized=finalize_probability_evaluation(labels,logits.numpy(),out/'evaluation');p=finalized['probabilities'];d=finalized['decisions'];rows=[];risk=[];classes=np.arange(p.shape[1]);mu=p@classes;sh=finalized['shrinkage']
 for rule,key,rkey in [('mode','mode_decision','mode_l1_risk'),('l1','l1_bayes_decision','l1_bayes_risk'),('l2','l2_bayes_decision','l2_bayes_risk')]:
  pred=d[key];err=np.abs(labels-pred);sev=err>=2;rows.append({'method':a.method,'decision':rule,'accuracy':float((pred==labels).mean()),'mae':float(err.mean()),'qwk':float(cohen_kappa_score(labels,pred,weights='quadratic')),'severe_count':int(sev.sum()),'severe_prevalence':float(sev.mean())});risk.append({'method':a.method,'decision':rule,'spearman':float(spearmanr(d[rkey],err).statistic),'severe_auroc':float(roc_auc_score(sev,d[rkey])),'severe_auprc':float(average_precision_score(sev,d[rkey]))})
 write(out/'decision_metrics.csv',rows);write(out/'risk_metrics.csv',risk); extreme=[]
 for true in (0,4):
  mask=labels==true;pred=d['l1_bayes_decision'];err=np.abs(labels[mask]-pred[mask]);row={'method':a.method,'true_class':true,'count':int(mask.sum()),'accuracy':float((pred[mask]==true).mean()),'mae':float(err.mean()),'severe_prevalence':float((err>=2).mean()),'mean_p_true':float(p[mask,true].mean()),'median_p_true':float(np.median(p[mask,true])),'mean_near_mass':float(p[mask,:2].sum(1).mean()) if true==0 else float(p[mask,3:].sum(1).mean()),'predictive_mean':float(mu[mask].mean()),'inward_shrinkage':float(sh[mask].mean()),'decision_bias':float((pred[mask]-true).mean()),'l1_risk':float(d['l1_bayes_risk'][mask].mean())};extreme.append(row)
  for target in range(5): row[f'pred_{target}_fraction']=float((pred[mask]==target).mean())
 write(out/'extreme_class_metrics.csv',extreme);torch.save({'model_state_dict':state,'method':a.method,'best_epoch':best_epoch},out/'best_checkpoint.pt');write(out/'training_history.csv',history);(out/'config.json').write_text(json.dumps({'method':a.method,'seed':a.seed,'epochs':a.epochs,'image_size':28,'batch_size':64,'optimizer':'AdamW','learning_rate':1e-3,'weight_decay':1e-4,'training_class_counts':counts.tolist(),'class_weights':weights.tolist() if a.method=='weighted_ce' else None,'slace_alpha':1.0 if a.method=='slace' else None,'checkpoint_selection':'minimum validation '+('weighted CE' if a.method=='weighted_ce' else 'SLACE')},indent=2)+'\n');(out/'summary.json').write_text(json.dumps({'best_epoch':best_epoch,'best_validation_score':best,'method':a.method},indent=2)+'\n')
if __name__=='__main__':main()
