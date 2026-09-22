#!/usr/bin/env python3
"""Fit frozen UTKFace C/N heads from validated feature archives only."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_utkface_prospective_replication import (CONSTRAINT_TOL, HEAD_BATCH, HEAD_EPOCHS, HEAD_LR, K, DirectionOnlyLinear, fit_direction_head, probabilities)
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage

REPLAY_TOL = 1e-4
EXPECTED_COUNTS = {'train':[2756,7128,2726,1210,404], 'validation':[459,1188,455,202,67], 'test':[459,1189,454,202,67]}

def sha256(path: Path) -> str:
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''): d.update(b)
 return d.hexdigest()

def write_json(path: Path, x: object) -> None: path.write_text(json.dumps(x,indent=2)+'\n')
def write_csv(path: Path, rows: list[dict]) -> None:
 with path.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def validate_feature_record(run: dict, metadata: dict) -> None:
 if run.get('readiness') != 'READY FOR HEAD ADAPTATION': raise RuntimeError('feature unit is not READY')
 if 'V100' not in str(run.get('execution',{}).get('gpu','')): raise RuntimeError('feature archive was not V100-produced')
 if metadata.get('feature_dimension') != 512: raise RuntimeError('feature dimension mismatch')
 if metadata.get('split_counts') != {'train':14224,'validation':2371,'test':2371}: raise RuntimeError('split count mismatch')
 if metadata.get('test_class_4_support') != 67: raise RuntimeError('test class-4 support mismatch')
 for r in metadata.get('replay',{}).values():
  if r.get('mode_differences') or r.get('l1_differences'): raise RuntimeError('source A replay decisions mismatch')

def load_archive(path: Path, split: str) -> dict[str,np.ndarray]:
 a=np.load(path/f'{split}_features.npz')
 out={k:a[k] for k in a.files}
 if out['features'].shape != (sum(EXPECTED_COUNTS[split]),512): raise RuntimeError(f'{split} feature shape mismatch')
 if np.bincount(out['labels'],minlength=K).tolist()!=EXPECTED_COUNTS[split]: raise RuntimeError(f'{split} class counts mismatch')
 if len(np.unique(out['sample_ids']))!=len(out['sample_ids']): raise RuntimeError(f'{split} duplicate IDs')
 if not all(np.isfinite(out[k]).all() for k in ('features','logits','probabilities')): raise RuntimeError(f'{split} non-finite values')
 return out

def constraint_audit(head: DirectionOnlyLinear, weight: torch.Tensor, bias: torch.Tensor) -> dict:
 return {'max_norm_error':float(head.max_norm_error().detach().cpu()),'max_bias_error':float((head.fixed_bias.detach().cpu()-bias.cpu()).abs().max()),'trainable_parameter_names':[n for n,p in head.named_parameters() if p.requires_grad]}

def condition_metrics(labels: np.ndarray, logits: np.ndarray) -> tuple[dict,dict[str,np.ndarray]]:
 p=probabilities(logits); d=bayes_decisions(p); yhat=d['l1_bayes_decision']; err=np.abs(labels-yhat); c4=labels==4; mean=p@np.arange(K); shrink=inward_shrinkage(labels,p)
 endpoint={'support':int(c4.sum()),'l1_mae':float(err[c4].mean()),'exact_match_rate':float((yhat[c4]==4).mean()),'mean_predictive_mean':float(mean[c4].mean()),'mean_inward_shrinkage':float(shrink[c4].mean()),'mean_p4':float(p[c4,4].mean()),'routing':[int((yhat[c4]==k).sum()) for k in range(K)]}
 classwise=[]
 for k in range(K):
  mask=labels==k
  classwise.append({'true_class':k,'support':int(mask.sum()),'mean_p4':float(p[mask,4].mean()) if mask.any() else float('nan'),'routing':[int((yhat[mask]==j).sum()) for j in range(K)]})
 metrics={'global_l1_mae':float(err.mean()),'macro_l1_mae':float(np.mean([err[labels==k].mean() for k in range(K) if (labels==k).any()])),'severe_error_rate_l1':float((err>=2).mean()),'endpoint_4':endpoint,'classwise_p4_and_routing':classwise}
 arrays={'logits':logits,'probabilities':p,'mode_decision':d['mode_decision'],'l1_bayes_decision':yhat,'l2_bayes_decision':d['l2_bayes_decision'],'l1_bayes_risk':d['l1_bayes_risk'],'predictive_mean':mean,'inward_shrinkage':shrink}
 return metrics,arrays

def run(args: argparse.Namespace) -> None:
 out=args.out
 if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
 if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name(0): raise RuntimeError('V100 CUDA allocation is required')
 fmanifest=json.loads(args.feature_manifest.read_text()); matches=[r for r in fmanifest['runs'] if r['loss']==args.objective and r['seed']==args.seed]
 if len(matches)!=1: raise RuntimeError('feature manifest unit missing or ambiguous')
 runrec=matches[0]; fdir=Path(runrec['archive_path']); meta=json.loads((fdir/'feature_export_metadata.json').read_text()); validate_feature_record(runrec,meta)
 if sha256(Path(runrec['checkpoint']))!=runrec['checkpoint_sha256'] or meta['checkpoint_sha256']!=runrec['checkpoint_sha256']: raise RuntimeError('source checkpoint hash mismatch')
 train,validation,test=(load_archive(fdir,s) for s in ('train','validation','test'))
 ids=[set(map(str,x['sample_ids'])) for x in (train,validation,test)]
 if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]: raise RuntimeError('feature split overlap')
 head_state=torch.load(fdir/'A_original_head.pt',map_location='cpu',weights_only=False); w,b=head_state['weight'].float(),head_state['bias'].float()
 if tuple(w.shape)!=(5,512) or tuple(b.shape)!=(5,): raise RuntimeError('A head shape mismatch')
 a_replay=float(np.abs(train['features']@w.numpy().T+b.numpy()-train['logits']).max())
 if a_replay>REPLAY_TOL: raise RuntimeError('A train feature replay mismatch')
 device=torch.device(args.device); xte=torch.as_tensor(test['features'],dtype=torch.float32,device=device)
 conditions={'A':test['logits']}; states={}; histories={}; audits={}
 for name,sampling in (('C','balanced'),('N','natural')):
  head,hist,draws,base=fit_direction_head(train['features'],train['labels'],w,b,sampling=sampling,seed=args.seed,device=device)
  audit={**base,**constraint_audit(head,w,b),'sampling':sampling,'objective':'ce','epochs':HEAD_EPOCHS,'batch_size':HEAD_BATCH,'learning_rate':HEAD_LR,'weight_decay':0.0,'observed_draw_counts':draws.tolist()}
  if audit['max_norm_error']>CONSTRAINT_TOL or audit['max_bias_error']>CONSTRAINT_TOL or audit['trainable_parameter_names']!=['direction']: raise RuntimeError(f'{name} constraint failure: {audit}')
  with torch.inference_mode(): conditions[name]=head(xte).cpu().numpy()
  states[name]=head.cpu().state_dict(); histories[name]=hist; audits[name]=audit
 metrics={}; arrays={}
 for name,logits in conditions.items(): metrics[name],arrays[name]=condition_metrics(test['labels'],logits)
 out.mkdir(parents=True)
 torch.save({'condition':'A_original','weight':w,'bias':b},out/'A_original_head.pt')
 for name in ('C','N'):
  torch.save({'condition':name,'state_dict':states[name],'fixed_norms':states[name]['fixed_norms'],'fixed_bias':states[name]['fixed_bias']},out/f'{name}_head.pt'); write_csv(out/f'{name}_history.csv',histories[name])
 payload={'sample_ids':test['sample_ids'],'labels':test['labels']}
 for name,a in arrays.items():
  for key,val in a.items(): payload[f'{name}_{key}']=val
 np.savez_compressed(out/'test_outputs.npz',**payload)
 rows=[]
 for name,m in metrics.items():
  e=m['endpoint_4'];rows.append({'condition':name,'endpoint_l1_mae':e['l1_mae'],'endpoint_exact_match':e['exact_match_rate'],'endpoint_mean_p4':e['mean_p4'],'endpoint_predictive_mean':e['mean_predictive_mean'],'endpoint_inward_shrinkage':e['mean_inward_shrinkage'],'endpoint_routing':json.dumps(e['routing']),'global_l1_mae':m['global_l1_mae'],'macro_l1_mae':m['macro_l1_mae'],'severe_error_rate_l1':m['severe_error_rate_l1']})
 write_csv(out/'per_condition_metrics.csv',rows)
 red=[]
 for k in range(K):
  a,n,c=(metrics[z]['classwise_p4_and_routing'][k] for z in ('A','N','C'));red.append({'true_class':k,'support':a['support'],'A_mean_p4':a['mean_p4'],'N_mean_p4':n['mean_p4'],'C_mean_p4':c['mean_p4'],'N_minus_A':n['mean_p4']-a['mean_p4'],'C_minus_A':c['mean_p4']-a['mean_p4'],'A_routing':json.dumps(a['routing']),'N_routing':json.dumps(n['routing']),'C_routing':json.dumps(c['routing'])})
 write_csv(out/'redistribution.csv',red)
 write_json(out/'manifest.json',{'objective':args.objective,'seed':args.seed,'feature_archive':str(fdir),'feature_manifest':str(args.feature_manifest),'checkpoint':runrec['checkpoint'],'checkpoint_sha256':runrec['checkpoint_sha256'],'a_train_replay_max_abs':a_replay,'constraints':audits,'metrics':metrics,'deltas':{'N_minus_A_endpoint_mae':metrics['N']['endpoint_4']['l1_mae']-metrics['A']['endpoint_4']['l1_mae'],'C_minus_A_endpoint_mae':metrics['C']['endpoint_4']['l1_mae']-metrics['A']['endpoint_4']['l1_mae'],'N_minus_C_endpoint_mae':metrics['N']['endpoint_4']['l1_mae']-metrics['C']['endpoint_4']['l1_mae']}})

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--objective',choices=('ce','rps'),required=True);p.add_argument('--seed',type=int,choices=(1,2,3,4),required=True);p.add_argument('--feature-manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');return p.parse_args()
if __name__=='__main__': run(parse_args())
