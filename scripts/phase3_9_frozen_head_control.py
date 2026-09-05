#!/usr/bin/env python3
"""Small Phase 3.9 frozen-linear-head control; never loads or updates a backbone."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ordinal_uncertainty.evaluation.frozen_head import inverse_frequency_weights, class_priors, balanced_ce_loss, logit_adjusted_ce_loss
from phase3_7a_solar_3ch import report
ROOT=Path('outputs/solar/phase3_9_mechanism_audit')
def loss(kind,z,y,w,p): return balanced_ce_loss(z,y,w) if kind=='balanced_ce' else logit_adjusted_ce_loss(z,y,p,1.)
def run(method,out,device):
 out.mkdir(parents=True,exist_ok=False); root=ROOT/'features'/method
 tr={k:v for k,v in np.load(root/'train.npz').items()}; va={k:v for k,v in np.load(root/'validation.npz').items()}; te={k:v for k,v in np.load(root/'test.npz').items()}
 pri=torch.tensor(class_priors(tr['labels'],5),dtype=torch.float32,device=device); wei=torch.tensor(inverse_frequency_weights(tr['labels'],5),dtype=torch.float32,device=device)
 results={}
 for kind in ('balanced_ce','logit_adjusted'):
  torch.manual_seed(0);np.random.seed(0);random.seed(0); head=nn.Linear(tr['features'].shape[1],5).to(device); opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4); dl=DataLoader(TensorDataset(torch.tensor(tr['features']),torch.tensor(tr['labels'])),batch_size=512,shuffle=True); best=float('inf');state=None;epoch0=0;wait=0;hist=[]
  for epoch in range(1,101):
   head.train()
   for x,y in dl: opt.zero_grad();v=loss(kind,head(x.to(device)),y.to(device),wei,pri);v.backward();opt.step()
   head.eval()
   with torch.no_grad(): score=float(torch.nn.functional.cross_entropy(head(torch.tensor(va['features'],device=device)),torch.tensor(va['labels'],device=device)).cpu())
   hist.append({'epoch':epoch,'validation_ce':score})
   if score<best: best=score;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};epoch0=epoch;wait=0
   else: wait+=1
   if wait>=10: break
  head.load_state_dict(state); z=head(torch.tensor(te['features'],device=device)).detach().cpu().numpy(); d=out/kind; result=report(te['labels'],z,te['sample_ids'],d/'evaluation');torch.save({'head_state_dict':state,'feature_source':method,'objective':kind,'frozen_backbone':True,'selected_epoch':epoch0,'validation_ce':best},d/'selected_head.pt');(d/'history.json').write_text(json.dumps(hist,indent=2)+'\n');results[kind]={'selected_epoch':epoch0,'validation_ce':best,'result':result}
 (out/'summary.json').write_text(json.dumps({'method':method,'frozen_backbone':True,'features_only':True,'controls':['balanced_ce','logit_adjusted'],'results':results},indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=('ce','rps'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.method,a.out,torch.device('cuda'))
if __name__=='__main__':main()
