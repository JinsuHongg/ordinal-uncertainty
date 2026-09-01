#!/usr/bin/env python3
"""Frozen Phase 3.7A solar CE/RPS replication: hmi_m, aia1600, aia131 only."""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

CHANNELS=("hmi_m","aia1600","aia131"); CHANNEL_INDICES=(8,7,1); EXPECTED=(
 (20130,18747,26832,8070,981),(761,1127,1384,376,24),(8346,6777,15810,11418,1497))
def label(v): return 0 if str(v).upper()=="FQ" else {"A":0,"B":1,"C":2,"M":3,"X":4}[str(v).upper()[0]]
def manifest(path, expected):
 d=pd.read_csv(path,parse_dates=['timestamp']); d['timestamp']=d.timestamp.astype('datetime64[ns]'); d['y']=d.max_goes_class.map(label); d['id']=d.timestamp.astype('int64')
 counts=tuple(int((d.y==i).sum()) for i in range(5))
 if counts!=expected: raise ValueError(f'{path}: counts {counts}, expected {expected}')
 return d.sort_values('timestamp').reset_index(drop=True)
def source_channels(root):
 import json
 a=json.loads((Path(root)/'2010/dataset/images/.zattrs').read_text())['channel_names']
 if tuple(a[i] for i in CHANNEL_INDICES)!=CHANNELS: raise ValueError(f'channel metadata mismatch: {a}')
 return a
class Solar(Dataset):
 def __init__(self, d, root, stats, augment=False):
  import zarr
  self.d=d; self.root=str(root); self.stats=np.asarray(stats[0],np.float32)[:,None,None],np.asarray(stats[1],np.float32)[:,None,None]; self.augment=augment; self.arr={}; self.pos={}
  for year in sorted({str(x.year) for x in d.timestamp}):
   g=zarr.open_group(f'{self.root}/{year}/dataset',mode='r'); t=np.asarray(g['time'][:],dtype='int64')-int(pd.Timedelta(hours=8).value); self.arr[year]=g['images']; self.pos[year]={int(v):i for i,v in enumerate(t)}
  keep=[]
  for _,r in d.iterrows():
   y=str(r.timestamp.year); keep.append(int(r.id) in self.pos.get(y,{}))
  self.d=d.loc[keep].reset_index(drop=True)
 def __len__(self): return len(self.d)
 def raw(self, i):
  r=self.d.iloc[i]; return np.asarray(self.arr[str(r.timestamp.year)][self.pos[str(r.timestamp.year)][int(r.id)]][list(CHANNEL_INDICES)],np.float32)
 def __getitem__(self,i):
  x=np.sign(self.raw(i))*np.log1p(np.abs(self.raw(i))); x=(x-self.stats[0])/self.stats[1]; x=torch.from_numpy(x)
  if self.augment and torch.rand(())<.5:x=x.flip(-1)
  if self.augment and torch.rand(())<.5:x=x.flip(-2)
  r=self.d.iloc[i]; return x,torch.tensor(int(r.y)),torch.tensor(int(r.id))
def rps(logits,y):
 p=torch.softmax(logits,1); oh=torch.nn.functional.one_hot(y,5).float(); return ((p.cumsum(1)[:,:-1]-oh.cumsum(1)[:,:-1])**2).sum(1).mean()/4
def decisions(p):
 c=np.arange(5); loss1=(p[:,None,:]*np.abs(c[None,None,:]-c[None,:,None])).sum(2); loss2=(p[:,None,:]*(c[None,None,:]-c[None,:,None])**2).sum(2); mode=p.argmax(1); return mode,loss1.argmin(1),loss2.argmin(1),loss1.min(1),loss1[np.arange(len(p)),mode]
def stat_dataset(d, root):
 # identity stats lets raw signed-log values be accessed; statistics use train only.
 return Solar(d,root,([0.,0.,0.],[1.,1.,1.]))
def statistics(ds):
 count=np.zeros(3); total=np.zeros(3); squared=np.zeros(3)
 for i in range(len(ds)):
  x=ds.raw(i).astype(np.float64); x=np.sign(x)*np.log1p(np.abs(x)); good=np.isfinite(x)
  count+=good.sum((1,2)); total+=np.where(good,x,0).sum((1,2)); squared+=np.where(good,x*x,0).sum((1,2))
 mean=total/count; std=np.sqrt(np.maximum(squared/count-mean*mean,1e-12))
 if not(np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()): raise ValueError('invalid train-only stats')
 return mean.tolist(),std.tolist()
def run(model,loader,loss,opt,dev):
 model.train(opt is not None); vals=[]
 for x,y,_ in loader:
  x,y=x.to(dev),y.to(dev)
  with torch.set_grad_enabled(opt is not None):
   v=loss(model(x),y)
   if opt: opt.zero_grad();v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
  vals.append(float(v.detach().cpu()))
 return float(np.mean(vals))
def report(y,z,ids,out):
 p=torch.softmax(torch.tensor(z),1).numpy(); mode,l1,l2,risk,mode_risk=decisions(p); one=np.eye(5)[y]; e=np.abs(y-l1); severe=e>=2
 def dm(d): return {'accuracy':float(accuracy_score(y,d)),'mae':float(np.abs(y-d).mean()),'qwk':float(cohen_kappa_score(y,d,weights='quadratic')),'severe_prevalence':float((np.abs(y-d)>=2).mean())}
 conf=p.max(1); bins=np.linspace(0,1,16); ece=sum(((conf>=bins[i])&((conf<bins[i+1]) if i<14 else (conf<=bins[i+1]))).mean()*abs((mode[(conf>=bins[i])&((conf<bins[i+1]) if i<14 else (conf<=bins[i+1]))]==y[(conf>=bins[i])&((conf<bins[i+1]) if i<14 else (conf<=bins[i+1]))]).mean()-conf[(conf>=bins[i])&((conf<bins[i+1]) if i<14 else (conf<=bins[i+1]))].mean()) for i in range(15) if ((conf>=bins[i])&((conf<bins[i+1]) if i<14 else (conf<=bins[i+1]))).any())
 order=np.argsort(risk); curve=[]
 for cov in np.arange(.1,1.01,.05):
  q=order[:math.ceil(cov*len(y))];curve.append({'coverage':float(cov),'ordinal_mae':float(e[q].mean())})
 def endpoint(k):
  m=y==k; pm=(p[m]*np.arange(5)).sum(1); adj=1 if k==0 else 3
  return {'count':int(m.sum()),'mode_routing':[int((mode[m]==a).sum()) for a in range(5)],'l1_routing':[int((l1[m]==a).sum()) for a in range(5)],'l2_routing':[int((l2[m]==a).sum()) for a in range(5)],'accuracy_l1':float((l1[m]==k).mean()),'mae_l1':float(e[m].mean()),'severe_l1':float(severe[m].mean()),'mean_p_true':float(p[m,k].mean()),'median_p_true':float(np.median(p[m,k])),'mean_p_adjacent':float(p[m,adj].mean()),'median_p_adjacent':float(np.median(p[m,adj])),'mean_top2':float((p[m,k]+p[m,adj]).mean()),'median_top2':float(np.median(p[m,k]+p[m,adj])),'predictive_mean':float(pm.mean()),'inward_shrinkage':float((k-pm).mean() if k else pm.mean()),'mean_l1_risk':float(risk[m].mean()),'median_l1_risk':float(np.median(risk[m]))}
 classes=[]
 for k in range(5):
  m=y==k; classes.append({'class':k,'count':int(m.sum()),'accuracy_l1':float((l1[m]==k).mean()),'mae_l1':float(e[m].mean()),'severe_l1':float(severe[m].mean()),'mean_true_probability':float(p[m,k].mean()),'predictive_mean':float((p[m]*np.arange(5)).sum(1).mean()),'mean_l1_risk':float(risk[m].mean())})
 auroc=None if severe.min()==severe.max() else float(roc_auc_score(severe,risk)); auprc=None if severe.min()==severe.max() else float(average_precision_score(severe,risk))
 out.mkdir(parents=True,exist_ok=False); np.savez_compressed(out/'predictions.npz',logits=z,probabilities=p,labels=y,sample_ids=ids,mode=mode,l1=l1,l2=l2,l1_bayes_risk=risk,mode_l1_risk=mode_risk)
 with (out/'risk_coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=curve[0]);w.writeheader();w.writerows(curve)
 result={'mode':dm(mode),'l1':dm(l1),'l2':dm(l2),'probability':{'nll':float(log_loss(y,p,labels=np.arange(5))),'brier':float(((p-one)**2).sum(1).mean()),'rps':float(((p.cumsum(1)[:,:-1]-one.cumsum(1)[:,:-1])**2).sum(1).mean()/4),'ece':float(ece)},'risk_quality':{'spearman':float(spearmanr(risk,e).statistic),'severe_auroc':auroc,'severe_auprc':auprc,'mean_selective_mae':float(np.mean([r['ordinal_mae'] for r in curve]))},'classwise':classes,'class0':endpoint(0),'x_class':endpoint(4)}
 (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); return result
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','smoke','train']);a.add_argument('--method',choices=['ce','rps']);a.add_argument('--root',default='/scratch/users/jhong36/data/surya-bench-224.zarr');a.add_argument('--index',default='/scratch/users/jhong36/data');a.add_argument('--out',required=True);a.add_argument('--stats');a.add_argument('--epochs',type=int,default=300);a.add_argument('--batch',type=int,default=16);a.add_argument('--workers',type=int,default=8);args=a.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=False); source_channels(args.root)
 splits=[manifest(Path(args.index)/x,e) for x,e in zip(('train.csv','validation.csv','test.csv'),EXPECTED)]
 if args.mode=='prepare':
  ds=stat_dataset(splits[0],args.root); mean,std=statistics(ds); payload={'channels':list(CHANNELS),'channel_indices':list(CHANNEL_INDICES),'shape':[3,224,224],'source_counts':[list(x) for x in EXPECTED],'image_aligned_train_count':len(ds),'mean':mean,'std':std,'transform':'sign(x)*log1p(abs(x))','fit_split':'train'}; (out/'normalization.json').write_text(json.dumps(payload,indent=2)+'\n');return
 if not args.stats: raise ValueError('--stats required')
 stats=json.loads(Path(args.stats).read_text());
 if tuple(stats['channels'])!=CHANNELS or tuple(stats['channel_indices'])!=CHANNEL_INDICES: raise ValueError('frozen channel metadata mismatch')
 ds=[Solar(d,args.root,(stats['mean'],stats['std']),augment=i==0) for i,d in enumerate(splits)]; loaders=[DataLoader(x,batch_size=args.batch,shuffle=i==0,num_workers=args.workers,pin_memory=True) for i,x in enumerate(ds)]
 torch.manual_seed(0);np.random.seed(0);random.seed(0); model=resnet18(weights=None); model.fc=torch.nn.Linear(512,5); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(dev);loss=torch.nn.CrossEntropyLoss() if args.method=='ce' else rps; opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
 if args.mode=='smoke':
  v=run(model,DataLoader(torch.utils.data.Subset(ds[0],range(min(2,len(ds[0])))),batch_size=2),loss,opt,dev);torch.save(model.state_dict(),out/'smoke_checkpoint.pt');(out/'smoke.json').write_text(json.dumps({'loss':v,'shape':list(ds[0][0][0].shape),'sample_id':int(ds[2][0][2]),'finite':bool(np.isfinite(v))})+'\n');return
 hist=[];best=float('inf');state=None;bestepoch=0;wait=0
 for epoch in range(1,args.epochs+1):
  tr=run(model,loaders[0],loss,opt,dev);va=run(model,loaders[1],loss,None,dev);hist.append({'epoch':epoch,'train_loss':tr,'validation_loss':va})
  if va<best:best=va;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestepoch=epoch;wait=0
  else:wait+=1
  if wait>=3:break
 model.load_state_dict(state);torch.save({'state_dict':state,'selected_epoch':bestepoch,'method':args.method,'seed':0},out/'selected_checkpoint.pt')
 z=[];y=[];ids=[];model.eval()
 with torch.no_grad():
  for x,t,i in loaders[2]:z.append(model(x.to(dev)).cpu());y.append(t);ids.append(i)
 result=report(torch.cat(y).numpy(),torch.cat(z).numpy(),torch.cat(ids).numpy(),out/'evaluation');
 with (out/'training_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
 (out/'config.json').write_text(json.dumps({'seed':0,'method':args.method,'channels':list(CHANNELS),'channel_indices':list(CHANNEL_INDICES),'stats':stats,'batch':args.batch,'epochs':args.epochs,'optimizer':'AdamW','learning_rate':5e-5,'weight_decay':.01,'checkpoint_selection':'minimum validation CE' if args.method=='ce' else 'minimum validation RPS','result':result},indent=2)+'\n')
if __name__=='__main__':main()
