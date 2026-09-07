#!/usr/bin/env python3
"""Frozen UTKFace A/B/C/D direction-scale confirmation; test is never loaded."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np,torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,cohen_kappa_score,roc_auc_score
from torch.utils.data import DataLoader
from ordinal_uncertainty.data.utkface import UTKFaceDataset,class_counts,load_manifest,records_for_split,utkface_transform
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error,prediction_metrics
from ordinal_uncertainty.models.resnet import make_resnet18
M=Path("/home/jhong90/github_proj/ordinal-aware-conformal/data/split_assignments/conference_v0_3/utkface/manifest.jsonl"); R=Path("/mnt/storage/data/utkface/UTKFace"); C=Path("outputs/utkface/phase3_7a_failure_replication/rps/seed_0/best_checkpoint.pt"); O=Path("outputs/utkface/phase3_13_direction_scale_mechanism_confirmation"); BS=64; E=100
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);ks=list(dict.fromkeys(k for r in rows for k in r))
 with p.open("w",newline="")as f:w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(rows)
def extract():
 train_cache,val_cache=O/"frozen_train_features.npz",O/"frozen_val_features.npz"
 if train_cache.exists()or val_cache.exists():raise FileExistsError("refusing to overwrite split feature cache")
 O.mkdir(parents=True,exist_ok=True);rs=load_manifest(M,R);sp={n:records_for_split(rs,n)for n in ("train","validation")}
 if class_counts(sp["train"])!=[2756,7128,2726,1210,404]or class_counts(sp["validation"])!=[459,1188,455,202,67]:raise ValueError("counts")
 if not torch.cuda.is_available()or torch.cuda.device_count()<1:raise RuntimeError("cuda:0 is mandatory")
 device=torch.device("cuda:0");s=torch.load(C,map_location="cpu",weights_only=False)["model_state_dict"];net=make_resnet18(5);net.load_state_dict(s);w,b=net.fc.weight.detach().clone(),net.fc.bias.detach().clone();net.fc=torch.nn.Identity();net.to(device).eval();out={}
 for n in ("train","validation"):
  f=[];y=[];i=[]
  with torch.inference_mode():
   for x,l,ids in DataLoader(UTKFaceDataset(sp[n],R,utkface_transform(False)),batch_size=BS,shuffle=False,num_workers=0):
    f.append(net(x.to(device,non_blocking=True)).cpu());y.append(l);i.extend(ids)
  out[n]=(torch.cat(f).numpy(),torch.cat(y).numpy(),np.asarray(i))
 for n,path in (("train",train_cache),("validation",val_cache)):
  tmp=path.with_name(path.stem+".tmp.npz");np.savez_compressed(tmp,features=out[n][0],labels=out[n][1],sample_ids=out[n][2],split=n,checkpoint=str(C),device="cuda:0",original_weight=w.numpy(),original_bias=b.numpy());tmp.replace(path)
 (O/"integrity.json").write_text(json.dumps({"manifest":str(M),"checkpoint":str(C),"device":"cuda:0","test_loaded":False,"counts":{n:class_counts(sp[n])for n in sp},"train_shape":list(out["train"][0].shape),"validation_shape":list(out["validation"][0].shape),"finite":all(np.isfinite(out[n][0]).all()for n in out),"id_overlap":int(len(set(out["train"][2])&set(out["validation"][2])))},indent=2)+"\n")
def p(z):q=np.exp(z-z.max(1,keepdims=True));return q/q.sum(1,keepdims=True)
def ep(y,pr,d,r,k):
 m=y==k;e=np.abs(d[m]-k);mu=pr[m]@np.arange(5);o={"routing":[int((d[m]==j).sum())for j in range(5)],"acc":float((d[m]==k).mean()),"mae":float(e.mean()),"severe":float((e>=2).mean()),"mean":float(mu.mean()),"risk":float(r[m].mean())}
 if k==4:o|={"p4":float(pr[m,4].mean()),"med_p4":float(np.median(pr[m,4])),"p3":float(pr[m,3].mean()),"med_p3":float(np.median(pr[m,3])),"p3p4":float(pr[m,3:].sum(1).mean()),"med_p3p4":float(np.median(pr[m,3:].sum(1))),"med_risk":float(np.median(r[m])),"shrinkage":float(4-mu.mean())}
 else:o|={"p0":float(pr[m,0].mean()),"p1":float(pr[m,1].mean()),"p0p1":float(pr[m,:2].sum(1).mean()),"med_risk":float(np.median(r[m]))}
 return o
def report(n,y,z):
 pr=p(z);q=bayes_decisions(pr);d=q["l1_bayes_decision"];e=np.abs(y-d);sv=e>=2;o=np.argsort(q["l1_bayes_risk"]);sel=[e[o[:max(1,int(np.ceil(c*len(y))))]].mean()for c in np.arange(1,.09,-.05)];g={}
 for n0,k in (("mode","mode_decision"),("l1","l1_bayes_decision"),("l2","l2_bayes_decision")):v=q[k];er=np.abs(y-v);g[n0]={"acc":float((v==y).mean()),"mae":float(er.mean()),"qwk":float(cohen_kappa_score(y,v,weights="quadratic")),"severe":float((er>=2).mean())}
 return {"global":g,"probability":prediction_metrics(y,pr)|{"ece":expected_calibration_error(y,pr)[0]},"risk":{"spearman":float(spearmanr(q["l1_bayes_risk"],e).statistic),"auroc":float(roc_auc_score(sv,q["l1_bayes_risk"])),"auprc":float(average_precision_score(sv,q["l1_bayes_risk"])),"selective_mae":float(np.mean(sel))},"c4":ep(y,pr,d,q["l1_bayes_risk"],4),"c0":ep(y,pr,d,q["l1_bayes_risk"],0)},pr,q
def fithead(h,x,y,wd):
 opt=torch.optim.AdamW(h.parameters()if isinstance(h,torch.nn.Linear)else[h.direction],lr=.001,weight_decay=wd);gen=torch.Generator().manual_seed(10000)
 history=[]
 for epoch in range(E):
  losses=[]
  for __ in range(int(np.ceil(len(y)/BS))):ix=balanced_batch_indices(y,BS,gen);loss=torch.nn.functional.cross_entropy(h(x[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  history.append({"epoch":epoch+1,"balanced_ce":float(np.mean(losses))})
 return history
def run():
 ta=np.load(O/"frozen_train_features.npz");va=np.load(O/"frozen_val_features.npz");torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);x=torch.tensor(ta["features"]);y=torch.tensor(ta["labels"]);vx=torch.tensor(va["features"]);vy=va["labels"];w=torch.tensor(ta["original_weight"]);b=torch.tensor(ta["original_bias"]);B=torch.nn.Linear(512,5);B.weight.data.copy_(w);B.bias.data.copy_(b);histories={"B":fithead(B,x,y,1e-4)};bw=B.weight.detach();H={"B":B};target=.5*w.norm(dim=1)+.5*bw.norm(dim=1)
 for n,norm in (("C",w.norm(dim=1)),("D",target)):
  h=DirectionOnlyLinear(w,b,norm);histories[n]=fithead(h,x,y,0);H[n]=h
  if float(h.max_norm_error().detach())>1e-6:raise RuntimeError("norm")
 states=O/"head_states";states.mkdir(parents=True,exist_ok=True)
 torch.save({"condition":"B","state_dict":B.state_dict(),"epochs":E,"optimizer":"AdamW","lr":.001,"weight_decay":1e-4},states/"B.pt")
 for n in ("C","D"):torch.save({"condition":n,"state_dict":H[n].state_dict(),"target_norms":H[n].fixed_norms.detach(),"fixed_bias":b,"epochs":E,"optimizer":"AdamW","lr":.001,"weight_decay":0.0},states/f"{n}.pt")
 for n,hist in histories.items():write(O/f"histories/{n}.csv",hist)
 z={"A":(vx@w.T+b).numpy()};res={};rows=[]
 for n,h in H.items():
  with torch.no_grad():z[n]=h(vx).numpy()
 for n,v in z.items():
  r,pr,q=report(n,vy,v);res[n]=r;write(O/f"predictions/{n}.csv",[{"sample_id":str(va["sample_ids"][i]),"label":int(vy[i]),"logits":json.dumps(v[i].tolist()),"probabilities":json.dumps(pr[i].tolist()),"mode":int(q["mode_decision"][i]),"l1":int(q["l1_bayes_decision"][i]),"l2":int(q["l2_bayes_decision"][i]),"risk":float(q["l1_bayes_risk"][i])}for i in range(len(vy))])
 for k in range(5):
  cos=lambda u,v:float(torch.dot(u,v)/(u.norm()*v.norm()))
  row={"class":k,"norm_a":float(w[k].norm()),"norm_b":float(bw[k].norm()),"norm_d":float(target[k]),"bias_a":float(b[k]),"bias_b":float(B.bias[k].detach()),"delta_bias":float(B.bias[k].detach()-b[k]),"cos_b_a":cos(bw[k],w[k])}
  for n in ("C","D"):row|={f"cos_{n.lower()}_a":cos(H[n].effective_weight()[k].detach(),w[k]),f"cos_{n.lower()}_b":cos(H[n].effective_weight()[k].detach(),bw[k]),f"norm_error_{n.lower()}":float(H[n].max_norm_error().detach())}
  rows.append(row)
 write(O/"parameters.csv",rows);(O/"summary.json").write_text(json.dumps({"protocol":{"alpha":.5,"rop":False,"test_loaded":False,"optimizer":"AdamW","lr":.001,"balanced_weight_decay":1e-4,"direction_weight_decay":0.0,"epochs":100,"batch":64},"results":res},indent=2)+"\n")
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("command",choices=("extract","run"));v=a.parse_args();extract()if v.command=="extract"else run()
