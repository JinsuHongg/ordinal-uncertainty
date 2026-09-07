#!/usr/bin/env python3
"""Read-only Phase 3.10B parameter/logit audit; never trains a model."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from ordinal_uncertainty.evaluation.head_audit import cosine_alignment, linear_logits, margin_decomposition, swapped_parameters
from ordinal_uncertainty.metrics.decision import bayes_decisions


P10A = Path("outputs/retinamnist/phase3_10a_rop_objective_falsification")
CHECKPOINT = Path("outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt")
OUT = Path("outputs/retinamnist/phase3_10b_head_bias_localization_audit")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True); values = np.exp(shifted); return values / values.sum(axis=1, keepdims=True)


def metric_row(name: str, labels: np.ndarray, logits: np.ndarray, fold: int | str) -> dict[str, object]:
    p = probabilities(logits); decision = bayes_decisions(p)["l1_bayes_decision"]; error = np.abs(labels - decision)
    c4, c0 = labels == 4, labels == 0
    return {"condition": name, "fold": fold, "global_accuracy_l1": float((decision == labels).mean()), "global_mae_l1": float(error.mean()), "global_qwk_l1": float(cohen_kappa_score(labels, decision, weights="quadratic")), "class4_mae_l1": float(error[c4].mean()), "class4_exact_l1": int((decision[c4] == 4).sum()), "class4_routing_l1": json.dumps([int((decision[c4] == k).sum()) for k in range(5)]), "class0_mae_l1": float(error[c0].mean())}


def summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9)), "fraction_positive": float((values > 0).mean())}


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    with np.load(P10A / "frozen_features/train_rps_features.npz") as x:
        ids, labels, features = x["sample_id"], x["labels"], x["features"]
    assignments = {int(row["sample_id"]): int(row["fold"]) for row in csv.DictReader((P10A / "fold_assignments/assignments.csv").open())}
    folds = np.asarray([assignments[int(i)] for i in ids])
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False); state = saved.get("model_state_dict", saved.get("state_dict"))
    original = {"weight": state["fc.weight"].numpy(), "bias": state["fc.bias"].numpy()}
    balanced = {}
    for fold in range(5):
        value = torch.load(P10A / "condition_b_balanced_head/fold_checkpoints" / f"fold_{fold}.pt", map_location="cpu", weights_only=False)["head_state_dict"]
        balanced[fold] = {"weight": value["weight"].numpy(), "bias": value["bias"].numpy()}
    if features.shape != (1080, 512) or not np.array_equal(ids, np.arange(1080)) or not np.array_equal(np.sort(folds), np.repeat(np.arange(5), [216] * 5)):
        raise ValueError("Phase 3.10A training-only alignment is invalid")
    return ids, labels, features, folds, original, balanced


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    ids, labels, features, folds, original, balanced = load(); OUT.mkdir(parents=True)
    for directory in ("parameters", "logits", "margins", "feature_alignment", "recovery_subsets", "class0_damage", "counterfactual_swaps", "rop_secondary", "summary"):
        (OUT / directory).mkdir(exist_ok=True)
    parameter_rows=[]; cosine_rows=[]; prior_rows=[]; all_logits={name:np.empty((len(ids),5)) for name in ("original", "balanced", "original_weights_balanced_biases", "balanced_weights_original_biases")}
    for fold in range(5):
        fit, held = folds != fold, folds == fold; bal = balanced[fold]
        priors=np.bincount(labels[fit], minlength=5)/fit.sum()
        for cls in range(5):
            ow,bw=original["weight"][cls],bal["weight"][cls]; on,bn=np.linalg.norm(ow),np.linalg.norm(bw)
            parameter_rows.append({"fold":fold,"class":cls,"log_train_prior":float(np.log(priors[cls])),"original_bias":float(original["bias"][cls]),"balanced_bias":float(bal["bias"][cls]),"delta_bias":float(bal["bias"][cls]-original["bias"][cls]),"original_weight_norm":float(on),"balanced_weight_norm":float(bn),"delta_weight_norm":float(bn-on),"direction_cosine":float(np.dot(ow,bw)/(on*bn)),"direction_angular_change_degrees":float(np.degrees(np.arccos(np.clip(np.dot(ow,bw)/(on*bn),-1,1))) )})
            prior_rows.append({"fold":fold,"class":cls,"fit_count":int((labels[fit]==cls).sum()),"prior":float(priors[cls]),"log_prior":float(np.log(priors[cls]))})
        for left in range(5):
            for right in range(left+1,5): cosine_rows.append({"fold":fold,"left_class":left,"right_class":right,"original_cosine":float(np.dot(original['weight'][left],original['weight'][right])/(np.linalg.norm(original['weight'][left])*np.linalg.norm(original['weight'][right]))),"balanced_cosine":float(np.dot(bal['weight'][left],bal['weight'][right])/(np.linalg.norm(bal['weight'][left])*np.linalg.norm(bal['weight'][right])))})
        for name, weight, bias in (("original",original["weight"],original["bias"]),("balanced",bal["weight"],bal["bias"]),("original_weights_balanced_biases",*swapped_parameters(original["weight"],original["bias"],bal["weight"],bal["bias"],"original_weights_balanced_biases")),("balanced_weights_original_biases",*swapped_parameters(original["weight"],original["bias"],bal["weight"],bal["bias"],"balanced_weights_original_biases"))):
            _, logits=linear_logits(features[held],weight,bias); all_logits[name][held]=logits
    write_csv(OUT/"parameters"/"per_fold_parameters.csv",parameter_rows); write_csv(OUT/"parameters"/"fold_priors.csv",prior_rows); write_csv(OUT/"parameters"/"pairwise_weight_cosines.csv",cosine_rows)
    table_a=[]
    for cls in range(5):
        r=[x for x in parameter_rows if x['class']==cls]; table_a.append({"class":cls,**{f"{k}_{s}":float(np.mean([x[k] for x in r])) if s=='mean' else float(np.std([x[k] for x in r],ddof=1)) for k in ('original_bias','balanced_bias','delta_bias','original_weight_norm','balanced_weight_norm','delta_weight_norm','direction_cosine') for s in ('mean','std')}})
    write_csv(OUT/"summary"/"table_a_head_parameters.csv",table_a)
    audit_rows=[]; margin_rows=[]; subset_rows=[]; sample_rows=[]
    for name in ("original","balanced"):
        weight=original['weight'] if name=='original' else None; bias=original['bias'] if name=='original' else None
        # B varies by fold; calculate every decomposition with its matched fold state.
        terms=np.empty((len(ids),5)); logits=all_logits[name]; align=np.empty((len(ids),5))
        for fold in range(5):
            held=folds==fold; w=original['weight'] if name=='original' else balanced[fold]['weight']; b=original['bias'] if name=='original' else balanced[fold]['bias']; terms[held], check=linear_logits(features[held],w,b); assert np.allclose(check,logits[held]); align[held]=cosine_alignment(features[held],w)
        p=probabilities(logits); decisions=bayes_decisions(p); audit_rows.extend([
            {"sample_id":int(ids[i]),"fold":int(folds[i]),"label":int(labels[i]),"condition":name,"logits":json.dumps(logits[i].tolist()),"feature_terms":json.dumps(terms[i].tolist()),"probabilities":json.dumps(p[i].tolist()),"l1_decision":int(decisions['l1_bayes_decision'][i]),**{f"alignment_{k}":float(align[i,k]) for k in range(5)}}
            for i in range(len(ids))
        ])
        for target,left,right in ((4,4,3),(4,4,2),(0,0,1)):
            mask=labels==target; feature,bias_term,total=margin_decomposition(terms,bias if name=='original' else np.zeros(5),left,right)
            # For B the bias is fold-specific, calculate explicitly.
            if name=='balanced':
                bias_vector=np.array([balanced[int(folds[i])]['bias'][left]-balanced[int(folds[i])]['bias'][right] for i in range(len(ids))]); total=feature+bias_vector
            else: bias_vector=np.full(len(ids),bias_term)
            stats=summary(total[mask]); margin_rows.append({"condition":name,"target_class":target,"margin":f"z{left}-z{right}",**stats,"mean_feature_term":float(feature[mask].mean()),"mean_bias_term":float(bias_vector[mask].mean())})
        np.savez_compressed(OUT/"feature_alignment"/f"{name}.npz", alignment=align)
    write_csv(OUT/"logits"/"oof_logit_decomposition.csv",audit_rows); write_csv(OUT/"margins"/"table_b_c_margins.csv",margin_rows)
    orig_dec=bayes_decisions(probabilities(all_logits['original']))['l1_bayes_decision']; bal_dec=bayes_decisions(probabilities(all_logits['balanced']))['l1_bayes_decision']
    for subset,mask,left,right in (("class4_exact_recovered",(labels==4)&(orig_dec!=4)&(bal_dec==4),4,3),("class4_inward_improved",(labels==4)&(np.abs(4-bal_dec)<np.abs(4-orig_dec)),4,3),("class0_damaged",(labels==0)&(np.abs(bal_dec)>np.abs(orig_dec)),0,1)):
        original_terms=[]; balanced_terms=[]; original_bias=[]; balanced_bias=[]
        for i in np.where(mask)[0]:
            f=int(folds[i]); ot,_=linear_logits(features[i:i+1],original['weight'],original['bias']); bt,_=linear_logits(features[i:i+1],balanced[f]['weight'],balanced[f]['bias']); original_terms.append(ot[0,left]-ot[0,right]); balanced_terms.append(bt[0,left]-bt[0,right]); original_bias.append(original['bias'][left]-original['bias'][right]); balanced_bias.append(balanced[f]['bias'][left]-balanced[f]['bias'][right])
            sample_rows.append({"subset":subset,"sample_id":int(ids[i]),"fold":f,"original_l1":int(orig_dec[i]),"balanced_l1":int(bal_dec[i]),"delta_feature_term":float(balanced_terms[-1]-original_terms[-1]),"delta_bias_term":float(balanced_bias[-1]-original_bias[-1]),"delta_total_margin":float((balanced_terms[-1]+balanced_bias[-1])-(original_terms[-1]+original_bias[-1]))})
        feature_delta = np.asarray(balanced_terms) - np.asarray(original_terms)
        bias_delta = np.asarray(balanced_bias) - np.asarray(original_bias)
        subset_rows.append({"subset":subset,"n":int(mask.sum()),"mean_delta_feature_term":float(feature_delta.mean()) if original_terms else 0.0,"mean_delta_bias_term":float(bias_delta.mean()) if original_bias else 0.0,"mean_delta_total_margin":float((feature_delta + bias_delta).mean()) if original_terms else 0.0})
    write_csv(OUT/"recovery_subsets"/"table_e_recovery_damage.csv",subset_rows); write_csv(OUT/"recovery_subsets"/"sample_margin_changes.csv",sample_rows)
    metrics=[]
    for name,logits in all_logits.items():
        metrics.append(metric_row(name,labels,logits,"pooled"))
        for fold in range(5): metrics.append(metric_row(name,labels[folds==fold],logits[folds==fold],fold))
    write_csv(OUT/"counterfactual_swaps"/"table_d_diagnostic_heads.csv",metrics)
    metadata={"training_performed":False,"validation_loaded":False,"test_loaded":False,"frozen_feature_source":str(P10A/'frozen_features/train_rps_features.npz'),"fold_assignment_source":str(P10A/'fold_assignments/assignments.csv'),"balanced_checkpoint_source":str(P10A/'condition_b_balanced_head/fold_checkpoints'),"checkpoint":str(CHECKPOINT),"feature_dimension":512,"samples":1080,"rop_secondary":"not run: no saved ROP head checkpoints and it is optional"}
    (OUT/"summary"/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__": main()
