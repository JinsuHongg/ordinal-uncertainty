#!/usr/bin/env python3
"""Phase 3.10C fixed-original-norm, fixed-bias direction-only OOF test."""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics

P10A=Path('outputs/retinamnist/phase3_10a_rop_objective_falsification')
CHECKPOINT=Path('outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt')
OUT=Path('outputs/retinamnist/phase3_10c_direction_only_head')
EPOCHS=100; BATCH=64

def write(path, rows):
    if not rows: raise ValueError('empty table')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def probs(z):
    z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def endpoint(y,p,d,c):
    m=y==c;e=np.abs(d[m]-c);mu=p[m]@np.arange(5);r=bayes_decisions(p)['l1_bayes_risk'][m]
    x={'support':int(m.sum()),'routing_l1':[int((d[m]==k).sum()) for k in range(5)],'mae':float(e.mean()),'exact':int((d[m]==c).sum()),'severe_prevalence':float((e>=2).mean()),'predictive_mean':float(mu.mean()),'mean_l1_risk':float(r.mean())}
    if c==4:x|={'mean_p4':float(p[m,4].mean()),'median_p4':float(np.median(p[m,4])),'mean_p3':float(p[m,3].mean()),'median_p3':float(np.median(p[m,3])),'mean_p3_p4':float(p[m,3:].sum(1).mean()),'inward_shrinkage':float(4-mu.mean())}
    else:x|={'mean_p0':float(p[m,0].mean()),'mean_p1':float(p[m,1].mean()),'mean_p0_p1':float(p[m,:2].sum(1).mean())}
    return x
def report(name,y,z,folds):
    p=probs(z);ds=bayes_decisions(p);d=ds['l1_bayes_decision'];e=np.abs(y-d);se=e>=2;order=np.argsort(ds['l1_bayes_risk']);cover=[e[order[:max(1,int(np.ceil(c*len(y))))]].mean() for c in np.arange(1,.09,-.05)]
    global_rows=[]
    for rule,key in [('mode','mode_decision'),('l1','l1_bayes_decision'),('l2','l2_bayes_decision')]:
        q=ds[key];err=np.abs(y-q);global_rows.append({'condition':name,'decision':rule,'accuracy':float((q==y).mean()),'mae':float(err.mean()),'qwk':float(cohen_kappa_score(y,q,weights='quadratic')),'severe_prevalence':float((err>=2).mean())})
    pr=prediction_metrics(y,p)|{'ece':expected_calibration_error(y,p)[0]}
    risk={'spearman':float(spearmanr(ds['l1_bayes_risk'],e).statistic),'severe_auroc':float(roc_auc_score(se,ds['l1_bayes_risk'])),'severe_auprc':float(average_precision_score(se,ds['l1_bayes_risk'])),'mean_selective_mae':float(np.mean(cover))}
    return p,ds,{'condition':name,'global':global_rows,'probability':pr,'risk':risk,'class4':endpoint(y,p,d,4),'class0':endpoint(y,p,d,0)}
def margins(y,z):
    out=[]
    for target,l,r in [(4,4,3),(4,4,2),(0,0,1)]:
        v=z[y==target,l]-z[y==target,r];out.append({'target':target,'margin':f'z{l}-z{r}','mean':float(v.mean()),'median':float(np.median(v)),'q10':float(np.quantile(v,.1)),'q90':float(np.quantile(v,.9)),'fraction_positive':float((v>0).mean())})
    return out
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if OUT.exists() and not args.resume:raise FileExistsError(f'refusing overwrite {OUT}')
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=args.resume)
    for directory in ('fold_states','oof_predictions','parameters','margins','recovery_subsets','safety_controls','summary'):(OUT/directory).mkdir(exist_ok=True)
    with np.load(P10A/'frozen_features/train_rps_features.npz') as x: ids,y,h=x['sample_id'],x['labels'],x['features']
    folds=np.asarray([int(r['fold']) for r in csv.DictReader((P10A/'fold_assignments/assignments.csv').open())])
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);state=saved.get('model_state_dict',saved.get('state_dict'));ow,ob=state['fc.weight'].float(),state['fc.bias'].float()
    z=np.empty((len(y),5),np.float32); histories=[];params=[];max_init=0.;max_norm=0.
    for fold in range(5):
        fit,held=folds!=fold,folds==fold;random.seed(fold);np.random.seed(fold);torch.manual_seed(fold)
        head=DirectionOnlyLinear(ow,ob); init=(head(torch.tensor(h[held])).detach()- (torch.tensor(h[held])@ow.T+ob)).abs().max().item();max_init=max(max_init,init)
        opt=torch.optim.AdamW([head.direction],lr=.001,weight_decay=0.0) # radial decay is removed by normalization
        generator=torch.Generator().manual_seed(10000+fold);x=torch.tensor(h[fit]);labels=torch.tensor(y[fit])
        for epoch in range(1,EPOCHS+1):
            losses=[]
            for _ in range(int(np.ceil(len(labels)/BATCH))):
                ix=balanced_batch_indices(labels,BATCH,generator);loss=torch.nn.functional.cross_entropy(head(x[ix]),labels[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
            error=float(head.max_norm_error().detach());max_norm=max(max_norm,error)
            if error>1e-6:raise RuntimeError(f'norm invariant failed fold {fold}: {error}')
            histories.append({'fold':fold,'epoch':epoch,'balanced_ce':float(np.mean(losses)),'max_norm_error':error})
        with torch.no_grad():z[held]=head(torch.tensor(h[held])).numpy();w=head.effective_weight().detach().numpy()
        torch.save({'head_state_dict':head.state_dict(),'fold':fold,'fixed_original_norms':ow.norm(dim=1).tolist(),'fixed_original_bias':ob.tolist(),'max_init_logit_error':init,'max_norm_error':float(head.max_norm_error().detach())},OUT/'fold_states'/f'fold_{fold}.pt')
        bw=torch.load(P10A/'condition_b_balanced_head/fold_checkpoints'/f'fold_{fold}.pt',map_location='cpu',weights_only=False)['head_state_dict']['weight'].numpy()
        for k in range(5):
            original_cosine=float(np.dot(w[k],ow[k].numpy())/(np.linalg.norm(w[k])*float(ow[k].norm())))
            balanced_cosine=float(np.dot(w[k],bw[k])/(np.linalg.norm(w[k])*np.linalg.norm(bw[k])))
            params.append({'fold':fold,'class':k,'fixed_norm':float(ow[k].norm()),'final_norm':float(np.linalg.norm(w[k])),'cosine_vs_original':original_cosine,'cosine_vs_balanced':balanced_cosine,'angle_from_original_degrees':float(np.degrees(np.arccos(np.clip(original_cosine,-1,1))) )})
    p,ds,c=report('C_direction_only',y,z,folds);write(OUT/'oof_predictions'/'predictions.csv',[{'sample_id':int(ids[i]),'fold':int(folds[i]),'label':int(y[i]),'logits':json.dumps(z[i].tolist()),'probabilities':json.dumps(p[i].tolist()),'l1_decision':int(ds['l1_bayes_decision'][i])}for i in range(len(y))]);write(OUT/'parameters'/'per_fold.csv',params);write(OUT/'summary'/'training.csv',histories);write(OUT/'margins'/'condition_c.csv',margins(y,z))
    # Reuse A/B stored OOF decisions for exact paired subset comparisons.
    def dload(n):return np.asarray([int(r['l1_decision']) for r in csv.DictReader((P10A/'oof_predictions'/n).open())])
    a,b=dload('A_original_rps.csv'),dload('B_balanced_head.csv');cd=ds['l1_bayes_decision'];c4=y==4;c0=y==0
    subsets=[{'name':'c4_exact','count':int((c4&(a!=4)&(cd==4)).sum()),'retained_from_b':int((c4&(a!=4)&(b==4)&(cd==4)).sum()),'b_total':int((c4&(a!=4)&(b==4)).sum())},{'name':'c4_inward','count':int((c4&(np.abs(4-cd)<np.abs(4-a))).sum()),'retained_from_b':int((c4&(np.abs(4-b)<np.abs(4-a))&(np.abs(4-cd)<np.abs(4-a))).sum()),'b_total':int((c4&(np.abs(4-b)<np.abs(4-a))).sum())},{'name':'c0_b_damage_restored','count':int((c0&(np.abs(cd)<np.abs(b))).sum()),'b_total':int((c0&(np.abs(b)>np.abs(a))).sum())}]
    write(OUT/'recovery_subsets'/'summary.csv',subsets)
    # A/B reports are frozen previous evidence; C is newly computed.
    a_summary=json.loads((P10A/'condition_a_original_rps/summary.json').read_text());b_summary=json.loads((P10A/'condition_b_balanced_head/summary.json').read_text());(OUT/'summary'/'summary.json').write_text(json.dumps({'A':a_summary,'B':b_summary,'C':c,'initial_logit_max_error':max_init,'max_norm_error':max_norm,'weight_decay':'0.0 on direction parameters; radial decay is functionally removed by normalization'},indent=2)+'\n')
if __name__=='__main__':main()
