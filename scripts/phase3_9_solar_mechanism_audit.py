#!/usr/bin/env python3
"""Phase 3.9 frozen solar feature extraction and train-centroid audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from ordinal_uncertainty.evaluation.representation import class_centroids, cosine_distances, euclidean_distances, l2_normalize, nearest_centroid, within_class_dispersion
from phase3_7a_solar_3ch import Solar, decisions, manifest, EXPECTED

ROOT=Path('outputs/solar/phase3_9_mechanism_audit')
P38=Path('outputs/solar/phase3_8_shrinkage_confirmation')
SPLITS=('train','validation','test')
def write(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')
def stats(): return json.loads((P38/'ce/seed_0/normalization_used.json').read_text())
def datasets():
 s=stats(); frames=[manifest(Path('/scratch/users/jhong36/data')/(x+'.csv'),e) for x,e in zip(SPLITS,EXPECTED)]
 ds=[Solar(f,'/scratch/users/jhong36/data/surya-bench-224.zarr',(s['mean'],s['std']),False) for f in frames]
 audit=json.loads((P38/'data_audit/alignment_audit.json').read_text())
 for n,d in zip(SPLITS,ds):
  if len(d)!=audit['splits'][n]['aligned_count']: raise ValueError('aligned subset mismatch '+n)
 return dict(zip(SPLITS,ds))
def model(checkpoint):
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False); state=saved['state_dict']
 m=resnet18(weights=None);m.fc=torch.nn.Linear(m.fc.in_features,5);m.load_state_dict(state,strict=True);return m
def extract(method,out,batch,workers):
 device=torch.device('cuda'); m=model(P38/method/'seed_0/selected_checkpoint.pt').to(device).eval(); data=datasets(); out.mkdir(parents=True,exist_ok=False)
 meta={'method':method,'checkpoint':str(P38/method/'seed_0/selected_checkpoint.pt'),'feature_layer':'input to model.fc after ResNet18 avgpool/flatten','feature_dimension':int(m.fc.in_features),'frozen_backbone':True,'backbone_updates':False,'splits':{}}
 for name,ds in data.items():
  feats=[]; logs=[]; labels=[]; ids=[]
  def hook(_m,inp): feats.append(inp[0].detach().cpu())
  h=m.fc.register_forward_pre_hook(hook)
  with torch.no_grad():
   for x,y,i in DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True): logs.append(m(x.to(device,non_blocking=True)).cpu());labels.append(y);ids.append(i)
  h.remove(); f=torch.cat(feats).numpy().astype('float32'); z=torch.cat(logs).numpy().astype('float32'); y=torch.cat(labels).numpy().astype('int64'); i=torch.cat(ids).numpy().astype('int64'); p=torch.softmax(torch.from_numpy(z),1).numpy(); mode,l1,l2,risk,_=decisions(p)
  np.savez_compressed(out/(name+'.npz'),sample_ids=i,labels=y,features=f,logits=z,probabilities=p,mode=mode,l1=l1,l2=l2,l1_bayes_risk=risk)
  meta['splits'][name]={'count':len(y),'class_counts':np.bincount(y,minlength=5).tolist(),'feature_shape':list(f.shape)}
 # strict Phase 3.8 test replay provenance
 a=np.load(out/'test.npz'); b=np.load(P38/method/'seed_0/evaluation/predictions.npz'); order={int(v):k for k,v in enumerate(b['sample_ids'])}; ix=np.array([order[int(v)] for v in a['sample_ids']]);
 if not(np.array_equal(a['labels'],b['labels'][ix]) and np.max(np.abs(a['logits']-b['logits'][ix]))<1e-5): raise ValueError('Phase 3.8 replay mismatch')
 meta['phase3_8_replay_max_abs_logit_error']=float(np.max(np.abs(a['logits']-b['logits'][ix])));write(out/'metadata.json',meta)
def summary(x): return {'mean':float(x.mean()),'median':float(np.median(x)),'fraction_positive':float((x>0).mean()),'fraction_negative':float((x<0).mean()),'min':float(x.min()),'max':float(x.max())}
def endpoint(test,nearest,dist,k):
 m=test['labels']==k; p=test['probabilities'][m]; dec={n:test[n][m] for n in ('mode','l1','l2')}; pm=(p*np.arange(5)).sum(1); err=np.abs(k-dec['l1']); own=dist[m,k]; adj=3 if k==4 else 1
 result={'count':int(m.sum()),'routing':[int((nearest[m]==j).sum()) for j in range(5)],'nearest_true_fraction':float((nearest[m]==k).mean()),'margin_vs_adjacent':summary(dist[m,adj]-own),'mode_routing':[int((dec['mode']==j).sum()) for j in range(5)],'l1_routing':[int((dec['l1']==j).sum()) for j in range(5)],'l2_routing':[int((dec['l2']==j).sum()) for j in range(5)],'mean_p_true':float(p[:,k].mean()),'mean_p_adjacent':float(p[:,adj].mean()),'mean_top2':float((p[:,k]+p[:,adj]).mean()),'predictive_mean':float(pm.mean()),'inward_shrinkage':float((k-pm).mean() if k else pm.mean()),'mean_l1_risk':float(test['l1_bayes_risk'][m].mean()),'mae_l1':float(err.mean()),'severe_l1':float((err>=2).mean())}
 if k==4:
  like=nearest[m]==4; result['groups']={}
  for n,g in [('representation_collapsed',~like),('representation_x_like',like)]:
   q=p[g]; dd={a:v[g] for a,v in dec.items()}; qq=(q*np.arange(5)).sum(1); ee=np.abs(4-dd['l1']); result['groups'][n]={'count':int(g.sum()),'mode_routing':[int((dd['mode']==j).sum()) for j in range(5)],'l1_routing':[int((dd['l1']==j).sum()) for j in range(5)],'l2_routing':[int((dd['l2']==j).sum()) for j in range(5)],'mean_p4':float(q[:,4].mean()),'mean_p3':float(q[:,3].mean()),'mean_p3p4':float((q[:,3]+q[:,4]).mean()),'predictive_mean':float(qq.mean()),'inward_shrinkage':float((4-qq).mean()),'mean_l1_risk':float(test['l1_bayes_risk'][m][g].mean()),'mae_l1':float(ee.mean()),'severe_l1':float((ee>=2).mean())}
  result['head_failure_rates']={a:float((dec[a][like]!=4).mean()) if like.any() else None for a in dec}
 return result
def geometry(out):
 out.mkdir(parents=True,exist_ok=False); allout={}
 for method in ('ce','rps'):
  root=ROOT/'features'/method; train={k:v for k,v in np.load(root/'train.npz').items()}; test={k:v for k,v in np.load(root/'test.npz').items()}
  result={'feature_dimension':int(train['features'].shape[1]),'spaces':{}}
  for name,feats,cent in [('raw_euclidean',train['features'],class_centroids(train['features'],train['labels'],5)),('l2_normalized_cosine',l2_normalize(train['features']),class_centroids(l2_normalize(train['features']),train['labels'],5))]:
   distfn=euclidean_distances if name=='raw_euclidean' else cosine_distances; td=distfn(test['features'] if name=='raw_euclidean' else l2_normalize(test['features']),cent); nearest=nearest_centroid(td); cd=distfn(cent,cent); disp=within_class_dispersion(feats,train['labels'],cent)
   result['spaces'][name]={'centroids':cent.tolist(),'centroid_distance_matrix':cd.tolist(),'within_class_dispersion':{k:v.tolist() for k,v in disp.items()},'test_nearest_accuracy_by_class':[float((nearest[test['labels']==k]==k).mean()) for k in range(5)],'x':endpoint(test,nearest,td,4),'class0':endpoint(test,nearest,td,0),'x_margins':{'delta_4_3':summary(td[test['labels']==4,3]-td[test['labels']==4,4]),'delta_4_2':summary(td[test['labels']==4,2]-td[test['labels']==4,4])}}
  allout[method]=result
 write(out/'geometry.json',allout)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('extract','geometry'));p.add_argument('--method',choices=('ce','rps'));p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=32);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 if a.mode=='extract':
  if not a.method: p.error('--method required')
  extract(a.method,a.out,a.batch,a.workers)
 else: geometry(a.out)
if __name__=='__main__':main()
