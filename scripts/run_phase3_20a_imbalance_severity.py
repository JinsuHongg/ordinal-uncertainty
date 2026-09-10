#!/usr/bin/env python3
"""Predeclared Phase 3.20A CE imbalance-severity dose-response grid."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import RetinaMNIST

from ordinal_uncertainty.evaluation.representation import class_centroids, cosine_distances, euclidean_distances, l2_normalize, nearest_centroid
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.utils.reproducibility import set_seed

ROOT = Path("outputs/retinamnist/phase3_20a_imbalance_severity_dose_response")
N4S = (66, 50, 33, 16, 8)
SEEDS = range(5)

class WithID(Dataset):
    def __init__(self, dataset, indices): self.dataset, self.indices = dataset, np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        image, label = self.dataset[int(self.indices[index])]
        return image, label.reshape(-1)[0], int(self.indices[index])

def write(path, rows):
    if not rows: raise ValueError(f"empty {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)

def loaders(seed, n4):
    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((28,28)), transforms.ToTensor(), transforms.Normalize((.5,)*3,(.5,)*3)])
    eval_tf = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor(), transforms.Normalize((.5,)*3,(.5,)*3)])
    raw = RetinaMNIST(split="train", root="data/medmnist", download=False)
    labels = raw.labels.reshape(-1).astype(np.int64)
    if np.bincount(labels, minlength=5).tolist() != [486,128,206,194,66]: raise RuntimeError("canonical train counts mismatch")
    rng = np.random.default_rng(seed); c4 = np.flatnonzero(labels == 4); retained = rng.permutation(c4)[:n4]
    indices = np.sort(np.concatenate([np.flatnonzero(labels != 4), retained]))
    if int((labels[indices] == 4).sum()) != n4: raise RuntimeError("invalid retained subset")
    train = WithID(RetinaMNIST(split="train", root="data/medmnist", download=False, transform=train_tf), indices)
    train_eval = WithID(RetinaMNIST(split="train", root="data/medmnist", download=False, transform=eval_tf), indices)
    val = WithID(RetinaMNIST(split="val", root="data/medmnist", download=False, transform=eval_tf), np.arange(120))
    test = WithID(RetinaMNIST(split="test", root="data/medmnist", download=False, transform=eval_tf), np.arange(400))
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train,64,shuffle=True,num_workers=0,generator=generator), DataLoader(train_eval,64,shuffle=False,num_workers=0), DataLoader(val,64,shuffle=False,num_workers=0), DataLoader(test,64,shuffle=False,num_workers=0)), labels, retained

def loss_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None; model.train(training); total=0.; count=0
    with torch.set_grad_enabled(training):
        for x,y,_ in loader:
            z=model(x.to(device)); loss=nn.functional.cross_entropy(z,y.long().to(device))
            if training: optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(y); count += len(y)
    return total/count

def infer(model, loader, device, features=False):
    ys=[]; zs=[]; ids=[]; hs=[]
    handle = model.fc.register_forward_pre_hook(lambda _m, inputs: hs.append(inputs[0].detach().cpu())) if features else None
    model.eval()
    with torch.no_grad():
        for x,y,i in loader: zs.append(model(x.to(device)).cpu()); ys.append(y); ids.append(i)
    if handle: handle.remove()
    result=(torch.cat(ids).numpy(), torch.cat(ys).numpy().astype(np.int64), torch.cat(zs).numpy().astype(np.float32))
    return result + ((torch.cat(hs).numpy().astype(np.float32),) if features else ())

def summarize(labels, logits, train_features, train_labels, test_features):
    p=torch.softmax(torch.from_numpy(logits),1).numpy(); d=bayes_decisions(p); l1=d['l1_bayes_decision']; err=np.abs(labels-l1); mu=p@np.arange(5)
    def endpoint(c):
        m=labels==c; e=err[m]; result={'support':int(m.sum()),'routing_l1':[int((l1[m]==k).sum()) for k in range(5)],'exact':int((l1[m]==c).sum()),'mae':float(e.mean()),'severe_prevalence':float((e>=2).mean()),'predictive_mean':float(mu[m].mean()),'mean_l1_risk':float(d['l1_bayes_risk'][m].mean()),'median_l1_risk':float(np.median(d['l1_bayes_risk'][m]))}
        if c==4: result.update(mean_p4=float(p[m,4].mean()),median_p4=float(np.median(p[m,4])),mean_p3=float(p[m,3].mean()),median_p3=float(np.median(p[m,3])),mean_p3_p4=float(p[m,3:].sum(1).mean()),median_p3_p4=float(np.median(p[m,3:].sum(1))),inward_shrinkage=float(4-mu[m].mean()))
        else: result.update(mean_p0=float(p[m,0].mean()),mean_p0_p1=float(p[m,:2].sum(1).mean()))
        return result
    c4=labels==4; margin=logits[c4,4]-logits[c4,3]
    cent=class_centroids(train_features,train_labels,5); raw=nearest_centroid(euclidean_distances(test_features,cent)); norm_cent=class_centroids(l2_normalize(train_features),train_labels,5); cosine=nearest_centroid(cosine_distances(test_features,norm_cent))
    global_l1={'accuracy':float((l1==labels).mean()),'mae':float(err.mean()),'qwk':float(cohen_kappa_score(labels,l1,weights='quadratic')),'severe_prevalence':float((err>=2).mean())}
    return {'class4':endpoint(4),'class0':endpoint(0),'global_l1':global_l1,'probability':prediction_metrics(labels,p)|{'ece':expected_calibration_error(labels,p)[0]},'margin_z4_z3':{'mean':float(margin.mean()),'median':float(np.median(margin)),'positive_fraction':float((margin>0).mean())},'geometry':{'raw_routing':[int((raw[c4]==k).sum()) for k in range(5)],'raw_feature_nearest_4_fraction':float((raw[c4]==4).mean()),'raw_representation_inward_fraction':float((raw[c4]!=4).mean()),'cosine_routing':[int((cosine[c4]==k).sum()) for k in range(5)],'cosine_feature_nearest_4_fraction':float((cosine[c4]==4).mean())}}, p, d

def run(seed,n4):
    out=ROOT/f"n4_{n4}"/f"seed_{seed}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); set_seed(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (train,train_eval,val,test), labels, retained=loaders(seed,n4)
    manifest={'seed':seed,'n4':n4,'retained_fraction':n4/66,'retained_class4_ids':retained.tolist(),'all_non4_ids':np.flatnonzero(labels!=4).tolist(),'canonical_train_counts':np.bincount(labels,minlength=5).tolist(),'subset_counts':np.bincount(labels[np.sort(np.r_[np.flatnonzero(labels!=4),retained])],minlength=5).tolist(),'validation_or_test_in_training':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    model=make_resnet18(5).to(device); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4); hist=[]; best=float('inf'); state=None; best_epoch=0
    for epoch in range(1,21):
        tr=loss_epoch(model,train,device,opt); va=loss_epoch(model,val,device)
        hist.append({'epoch':epoch,'train_ce':tr,'validation_ce':va})
        if va<best: best=va; best_epoch=epoch; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); train_id,train_y,_,train_h=infer(model,train_eval,device,True); test_id,test_y,test_z,test_h=infer(model,test,device,True)
    summary,p,d=summarize(test_y,test_z,train_h,train_y,test_h)
    torch.save({'model_state_dict':state,'seed':seed,'n4':n4,'best_epoch':best_epoch,'best_validation_nll':best},out/'best_checkpoint.pt')
    np.savez_compressed(out/'evaluation.npz',test_sample_id=test_id,test_labels=test_y,test_logits=test_z,test_probabilities=p,l1_decision=d['l1_bayes_decision'],train_sample_id=train_id,train_labels=train_y,train_features=train_h,test_features=test_h)
    write(out/'training_history.csv',hist); (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (out/'config.json').write_text(json.dumps({'seed':seed,'n4':n4,'epochs':20,'batch_size':64,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'loss':'cross_entropy','checkpoint_selection':'minimum validation NLL','architecture':'unpretrained ResNet18; 3x3 stride-1 stem; no maxpool','preprocessing':'Resize(28), ToTensor, Normalize(.5,.5,.5)','augmentation':'RandomHorizontalFlip training only','device':str(device)},indent=2)+'\n')

def aggregate():
    rows=[]
    for n4 in N4S:
        for seed in SEEDS:
            p=ROOT/f"n4_{n4}"/f"seed_{seed}"/'summary.json'
            if not p.exists(): raise FileNotFoundError(p)
            x=json.loads(p.read_text()); c=x['class4']; rows.append({'seed':seed,'n4':n4,'retained_fraction':n4/66,'c4_mae':c['mae'],'shrinkage':c['inward_shrinkage'],'predictive_mean':c['predictive_mean'],'mean_p4':c['mean_p4'],'severe_c4':c['severe_prevalence'],'margin_z4_z3':x['margin_z4_z3']['mean'],'margin_positive_fraction':x['margin_z4_z3']['positive_fraction'],'feature_nearest_4_fraction':x['geometry']['raw_feature_nearest_4_fraction'],'c0_mae':x['class0']['mae'],'global_l1_mae':x['global_l1']['mae']})
    write(ROOT/'analysis/per_seed_severity_metrics.csv',rows); metrics=[k for k in rows[0] if k not in ('seed','n4','retained_fraction')]; aggregate_rows=[]; trend=[]
    for n4 in N4S:
        subset=[r for r in rows if r['n4']==n4]
        aggregate_rows.append({'n4':n4,'retained_fraction':n4/66,**{f'{m}_mean':float(np.mean([r[m] for r in subset])) for m in metrics},**{f'{m}_std':float(np.std([r[m] for r in subset],ddof=1)) for m in metrics}})
    for m in metrics:
        means=[r[f'{m}_mean'] for r in aggregate_rows]; trend.append({'metric':m,'aggregate_spearman_n4':float(spearmanr(N4S,means).statistic),'run_level_spearman_n4':float(spearmanr([r['n4'] for r in rows],[r[m] for r in rows]).statistic)})
        for seed in SEEDS: trend.append({'metric':m,'seed':seed,'slope_vs_log_n4':float(np.polyfit(np.log(N4S),[r[m] for r in rows if r['seed']==seed],1)[0])})
    write(ROOT/'analysis/severity_mean_std.csv',aggregate_rows); write(ROOT/'analysis/trends.csv',trend)
    (ROOT/'analysis/completeness.json').write_text(json.dumps({'expected_runs':25,'observed_runs':len(rows),'cells':[(s,n) for s in SEEDS for n in N4S]},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--n4',type=int,choices=N4S);p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.aggregate: aggregate()
    elif a.seed is not None and a.n4 is not None: run(a.seed,a.n4)
    else: raise ValueError('supply --seed/--n4 or --aggregate')
if __name__=='__main__': main()
