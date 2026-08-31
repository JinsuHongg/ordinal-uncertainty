#!/usr/bin/env python3
"""Seed-0 smoke/full CORAL or RPS training; no multi-seed launcher."""
from __future__ import annotations
import argparse,sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.models.ordinal import coral_loss,coral_probabilities,coral_prediction,rps_loss
from ordinal_uncertainty.evaluation.ordinal_uncertainty import evaluate_predictions
from ordinal_uncertainty.utils.reproducibility import set_seed
def main():
 ap=argparse.ArgumentParser();ap.add_argument('method',choices=['coral','rps']);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);set_seed(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');loaders,meta=retinamnist_loaders(Path('data/medmnist'),64,0,False,28); model=make_resnet18(meta['num_classes']).to(device)
 if a.method=='coral': model.fc=nn.Linear(model.fc.in_features,1,bias=False).to(device);model.coral_bias=nn.Parameter(torch.linspace(1,-1,4,device=device));
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);history=[];best=float('inf');best_state=None;best_epoch=0
 for epoch in range(a.epochs):
  model.train();losses=[]
  for x,y in loaders['train']:
   x,y=x.to(device),y.reshape(-1).long().to(device);z=model(x); z=z+model.coral_bias if a.method=='coral' else z;loss=coral_loss(z,y) if a.method=='coral' else rps_loss(z,y);opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  model.eval();vl=[]
  with torch.no_grad():
   for x,y in loaders['val']:
    z=model(x.to(device));z=z+model.coral_bias if a.method=='coral' else z;vl.append((coral_loss(z,y.reshape(-1).long().to(device)) if a.method=='coral' else rps_loss(z,y.reshape(-1).long().to(device))).item())
  val=sum(vl)/len(vl);history.append({'epoch':epoch+1,'train_loss':sum(losses)/len(losses),'val_loss':val})
  if val<best: best,best_epoch,best_state=val,epoch+1,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(best_state);torch.save({'state_dict':best_state,'method':a.method,'best_epoch':best_epoch},out/'best_checkpoint.pt')
 model.eval();ps=[];ys=[];zs=[]
 with torch.no_grad():
  for x,y in loaders['test']:
   z=model(x.to(device));z=z+model.coral_bias if a.method=='coral' else z;zs.append(z.cpu());ys.append(y.reshape(-1));ps.append(torch.sigmoid(z).cpu() if a.method=='coral' else torch.softmax(z,1).cpu())
 q=torch.cat(ps);labels=torch.cat(ys); logits=torch.cat(zs);diag={'history':history,'method':a.method,'best_epoch':best_epoch,'best_validation_score':best,'selection_metric':'validation CORAL loss' if a.method=='coral' else 'validation RPS'}
 if a.method=='coral':
  d=q[:,:-1]-q[:,1:];diag|={'biases':model.coral_bias.detach().cpu().tolist(),'nonmonotone_count':int((d < 0).any(1).sum()),'max_violation':float((-d).clamp_min(0).max()),'min_adjacent_difference':float(d.min())}; probs=coral_probabilities(logits)
 else: diag|={'min_probability':float(q.min()),'max_sum_error':float((q.sum(1)-1).abs().max())};probs=q
 evaluation_logits = torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny)) if a.method=='coral' else logits
 evaluate_predictions(labels.numpy(),evaluation_logits.numpy(),probs.numpy(),out/'evaluation')
 with (out/'training_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
 (out/'config.json').write_text(json.dumps({'seed':a.seed,'method':a.method,'epochs':a.epochs,'image_size':28,'batch_size':64,'optimizer':'AdamW','learning_rate':1e-3,'weight_decay':1e-4,'checkpoint_selection':diag['selection_metric']},indent=2)+'\n')
 (out/'smoke.json').write_text(json.dumps(diag,indent=2)+'\n')
if __name__=='__main__':main()
