#!/usr/bin/env python3
"""Phase 3.10D fixed interpolated-scale direction-only OOF falsification."""
from __future__ import annotations
import argparse,csv,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,cohen_kappa_score,roc_auc_score
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error,prediction_metrics
P=Path('outputs/retinamnist/phase3_10a_rop_objective_falsification');C=Path('outputs/retinamnist/phase3_10c_direction_only_head');CK=Path('outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt');OUT=Path('outputs/retinamnist/phase3_10d_controlled_scale_head');ALPHAS=(.25,.5,.75);E=100;BS=64
def write(p,r):
 p.parent.mkdir(parents=True,exist_ok=True);ks=list(dict.fromkeys(k for x in r for k in x));
 with p.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(r)
def prob(z):z=z-z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)
def ep(y,p,d,k):
 m=y==k;e=np.abs(d[m]-k);mu=p[m]@np.arange(5);r=bayes_decisions(p)['l1_bayes_risk'][m];x={'routing':[int((d[m]==j).sum())for j in range(5)],'mae':float(e.mean()),'exact':int((d[m]==k).sum()),'severe':float((e>=2).mean()),'predictive_mean':float(mu.mean()),'risk':float(r.mean())}
 if k==4:x|={'p4':float(p[m,4].mean()),'p3':float(p[m,3].mean()),'p3p4':float(p[m,3:].sum(1).mean()),'shrinkage':float(4-mu.mean())}
 else:x|={'p0':float(p[m,0].mean()),'p1':float(p[m,1].mean()),'p0p1':float(p[m,:2].sum(1).mean())}
 return x
def summarize(name,y,z):
 p=prob(z);dct=bayes_decisions(p);d=dct['l1_bayes_decision'];e=np.abs(y-d);sev=e>=2;order=np.argsort(dct['l1_bayes_risk']);sel=[e[order[:max(1,int(np.ceil(c*len(y))))]].mean()for c in np.arange(1,.09,-.05)]
 dec=[]
 for n,k in [('mode','mode_decision'),('l1','l1_bayes_decision'),('l2','l2_bayes_decision')]:
  q=dct[k];ee=np.abs(y-q);dec.append({'rule':n,'accuracy':float((q==y).mean()),'mae':float(ee.mean()),'qwk':float(cohen_kappa_score(y,q,weights='quadratic')),'severe':float((ee>=2).mean())})
 margins=[]
 for t,l,r in [(4,4,3),(4,4,2),(0,0,1)]:
  v=z[y==t,l]-z[y==t,r];margins.append({'target':t,'margin':f'z{l}-z{r}','mean':float(v.mean()),'median':float(np.median(v)),'q10':float(np.quantile(v,.1)),'q90':float(np.quantile(v,.9)),'positive':float((v>0).mean())})
 return p,d,{'condition':name,'decisions':dec,'probability':prediction_metrics(y,p)|{'ece':expected_calibration_error(y,p)[0]},'risk':{'spearman':float(spearmanr(dct['l1_bayes_risk'],e).statistic),'auroc':float(roc_auc_score(sev,dct['l1_bayes_risk'])),'auprc':float(average_precision_score(sev,dct['l1_bayes_risk'])),'selective_mae':float(np.mean(sel))},'c4':ep(y,p,d,4),'c0':ep(y,p,d,0),'margins':margins}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 if OUT.exists() and not args.resume:raise FileExistsError(OUT)
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=args.resume)
 with np.load(P/'frozen_features/train_rps_features.npz')as x:ids,y,h=x['sample_id'],x['labels'],x['features']
 folds=np.asarray([int(r['fold'])for r in csv.DictReader((P/'fold_assignments/assignments.csv').open())]);s=torch.load(CK,map_location='cpu',weights_only=False);st=s.get('model_state_dict',s.get('state_dict'));ow,ob=st['fc.weight'].float(),st['fc.bias'].float();results={};all_params=[];all_history=[];all_sub=[]
 a=np.asarray([int(r['l1_decision'])for r in csv.DictReader((P/'oof_predictions/A_original_rps.csv').open())]);b=np.asarray([int(r['l1_decision'])for r in csv.DictReader((P/'oof_predictions/B_balanced_head.csv').open())])
 for alpha in ALPHAS:
  alpha_path=OUT/'oof_predictions'/f'alpha_{str(alpha).replace(".","p")}.csv'
  if args.resume and alpha_path.exists():
   saved_rows=list(csv.DictReader(alpha_path.open()));z=np.asarray([json.loads(r['logits'])for r in saved_rows],dtype=np.float32);p,d,res=summarize(f'D_alpha_{alpha}',y,z);c4=y==4;c0=y==0;exact=(c4&(a!=4)&(b==4));inward=(c4&(np.abs(4-b)<np.abs(4-a)));dam=(c0&(np.abs(b)>np.abs(a)));all_sub.append({'alpha':alpha,'b_exact_total':int(exact.sum()),'exact_retained':int((exact&(d==4)).sum()),'b_inward_total':int(inward.sum()),'inward_retained':int((inward&(np.abs(4-d)<np.abs(4-a))).sum()),'b_c0_damaged':int(dam.sum()),'c0_restored':int((dam&(np.abs(d)<np.abs(b))).sum())});res|={'alpha':alpha,'initial_logit_error':0.0,'max_norm_error':0.0};results[str(alpha)]=res;continue
  z=np.empty((len(y),5),np.float32);maxerr=0.;initerr=0.;state_dir=OUT/'fold_states'/f'alpha_{str(alpha).replace(".","p")}';state_dir.mkdir(parents=True)
  for fold in range(5):
   fit,held=folds!=fold,folds==fold;random.seed(fold);np.random.seed(fold);torch.manual_seed(fold);bw=torch.load(P/'condition_b_balanced_head/fold_checkpoints'/f'fold_{fold}.pt',map_location='cpu',weights_only=False)['head_state_dict']['weight'].float();norm=(1-alpha)*ow.norm(dim=1)+alpha*bw.norm(dim=1);head=DirectionOnlyLinear(ow,ob,norm);initerr=max(initerr,float((head(torch.tensor(h[held]))-(torch.tensor(h[held])@head.effective_weight().T+ob)).abs().max().detach()))
   opt=torch.optim.AdamW([head.direction],lr=.001,weight_decay=0.0);g=torch.Generator().manual_seed(10000+fold);x=torch.tensor(h[fit]);lab=torch.tensor(y[fit])
   for epoch in range(1,E+1):
    ls=[]
    for _ in range(int(np.ceil(len(lab)/BS))):
     ix=balanced_batch_indices(lab,BS,g);loss=torch.nn.functional.cross_entropy(head(x[ix]),lab[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls.append(float(loss.detach()))
    er=float(head.max_norm_error().detach());maxerr=max(maxerr,er)
    if er>1e-6:raise RuntimeError(er)
    all_history.append({'alpha':alpha,'fold':fold,'epoch':epoch,'loss':float(np.mean(ls)),'norm_error':er})
   with torch.no_grad():z[held]=head(torch.tensor(h[held])).numpy();w=head.effective_weight().detach().numpy()
   torch.save({'head_state_dict':head.state_dict(),'alpha':alpha,'fold':fold,'target_norms':norm.tolist(),'max_norm_error':float(head.max_norm_error().detach())},state_dir/f'fold_{fold}.pt')
   for k in range(5):all_params.append({'alpha':alpha,'fold':fold,'class':k,'target_norm':float(norm[k]),'final_norm':float(np.linalg.norm(w[k])),'norm_error':float(abs(np.linalg.norm(w[k])-norm[k])),'cosine_a':float(np.dot(w[k],ow[k].numpy())/(np.linalg.norm(w[k])*float(ow[k].norm()))),'cosine_b':float(np.dot(w[k],bw[k].numpy())/(np.linalg.norm(w[k])*float(bw[k].norm())))})
  p,d,res=summarize(f'D_alpha_{alpha}',y,z);c4=y==4;c0=y==0;exact=(c4&(a!=4)&(b==4));inward=(c4&(np.abs(4-b)<np.abs(4-a)));dam=(c0&(np.abs(b)>np.abs(a)));all_sub.append({'alpha':alpha,'b_exact_total':int(exact.sum()),'exact_retained':int((exact&(d==4)).sum()),'b_inward_total':int(inward.sum()),'inward_retained':int((inward&(np.abs(4-d)<np.abs(4-a))).sum()),'b_c0_damaged':int(dam.sum()),'c0_restored':int((dam&(np.abs(d)<np.abs(b))).sum())});res|={'alpha':alpha,'initial_logit_error':initerr,'max_norm_error':maxerr};results[str(alpha)]=res;write(OUT/'oof_predictions'/f'alpha_{str(alpha).replace(".","p")}.csv',[{'sample_id':int(ids[i]),'fold':int(folds[i]),'label':int(y[i]),'logits':json.dumps(z[i].tolist()),'probabilities':json.dumps(p[i].tolist()),'mode':int(bayes_decisions(p)['mode_decision'][i]),'l1':int(d[i]),'l2':int(bayes_decisions(p)['l2_bayes_decision'][i]),'risk':float(bayes_decisions(p)['l1_bayes_risk'][i]),'alpha':alpha}for i in range(len(y))]);write(OUT/'margins'/f'alpha_{str(alpha).replace(".","p")}.csv',res['margins'])
 write(OUT/'parameters'/'per_fold.csv',all_params);write(OUT/'summary'/'training.csv',all_history);write(OUT/'recovery_subsets'/'retention.csv',all_sub);(OUT/'summary'/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
