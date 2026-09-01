# Phase 3.6 — RG-ACR Seed-0 Falsification

## Scope and hypothesis

This is the single predeclared final RetinaMNIST seed-0 method-selection experiment. The required project-state documents, Phase 3.2–3.5 notes, RPS implementation, representation utilities, and frozen-head utilities were reviewed before code changes.

The hypothesis was that detached ordinal L1 Bayes risk can guide adjacent-class representation separation, localizing rare high-risk endpoints while preserving RPS risk geometry. Only the Phase 3.5 RG-ACR specification was implemented. No RG-ACR-v2, other dataset, or seeds 1–4 were created.

The frozen historical RPS artifacts are absent from this checkout because outputs are ignored. A canonical matched seed-0 RPS reference was retrained solely for strict comparison; it is not a replacement claim for the prior historical RPS evidence.

## Frozen objective and protocol

Canonical native-28 RetinaMNIST, official splits, unpretrained small-image ResNet18, 3x3 stride-1 stem, no max-pool, seed 0, batch 64, AdamW learning rate .001, weight decay .0001, and 20 epochs were used. Checkpoint selection is minimum validation RPS only.

For penultimate feature \(h_i\in\mathbb R^{512}\), \(z_i=h_i/(\|h_i\|_2+10^{-8})\), and \(d(u,v)=1-u^\top v\). The own-class centroid is leave-one-out; an anchor is valid iff its class count is at least two and one or more adjacent class is present. Missing adjacent classes are skipped; empty valid-anchor batches yield zero representation loss.

\[
R_i=\min_a\sum_kp_{ik}|k-a|,\qquad
w_i=\min\left(2,\frac{\operatorname{sg}(R_i)}
{\operatorname{mean}_{r\in B}\operatorname{sg}(R_r)+10^{-8}}\right).
\]

\[
\ell_i=\frac{1}{|A_B(y_i)|}\sum_{j\in A_B(y_i)}
[.05+d(z_i,c_{y_i,-i})-d(z_i,c_j)]_+,
\]

\[
L_{\rm RG-ACR}=\frac{\sum_{i\in V_B}w_i\ell_i}{\sum_{i\in V_B}w_i+10^{-8}},
\qquad L=L_{\rm RPS}+\lambda L_{\rm RG-ACR}.
\]

Risk is detached. Only \(\lambda\in\{.05,.10,.20\}\) was evaluated; no class weighting, resampling, non-adjacent negative, joint head adjustment, or other auxiliary term was added.

## Smoke and batch participation

The one-epoch λ=.10 smoke had finite RPS and RG-ACR loss, finite backbone gradient, an active feature-gradient path, no zero-loss batch, and all 66 class-4 examples valid. It did not evaluate test performance.

Each full RG-ACR run had 340 batches: overall valid-anchor rate 99.88%; class-4 valid anchors 1295/1320 (98.11%); mean/median class-4 valid anchors per batch 3.81/4; batches with class 4 present 98.53%; batches with valid class-4 anchor 91.18%; class-3 availability for class-4 anchors 98.11%; and zero representation-loss batches 0%. Inadequate target-class participation does not explain the result.

## Validation-only selection

| Condition | Selected epoch | Validation RPS |
| --- | ---: | ---: |
| Retrained RPS reference | 6 | .11855 |
| RG-ACR λ=.05 | 5 | **.11461** |
| RG-ACR λ=.10 | 10 | .11710 |
| RG-ACR λ=.20 | 3 | .11775 |

λ=.05 is the sole selected condition. Test results from the other λ values were not used for selection.

## Representation test

Raw geometry is Euclidean and normalized geometry is cosine; both use train-derived centroids. Values below are class-4 test diagnostics.

| Condition | Raw 4→4 | Raw Δ4,3 | Raw Δ4,2 | Raw D3,4 | Norm 4→4 | Norm Δ4,3 | Norm Δ4,2 | Norm D3,4 | Raw C4 dispersion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RPS reference | 8 | -.096 | .087 | .753 | 9 | -.0041 | -.0075 | .001016 | 6.609 |
| λ=.05 selected | 7 | -.037 | .081 | .697 | 7 | -.0031 | -.0054 | .001011 | 4.928 |
| λ=.10 | 3 | -.892 | -1.152 | 1.960 | 4 | -.0325 | -.0596 | .006615 | 10.332 |
| λ=.20 | 12 | .141 | .471 | .674 | 12 | .0010 | .0050 | .000900 | 5.382 |

All conditions retained descriptive class-4 centroid ordering \(D_{4,3}<D_{4,2}<D_{4,1}<D_{4,0}\). For selected λ=.05, raw Δ4,3 is less negative, but class-4 routing falls in both spaces, raw Δ4,2 falls, and D3,4 falls in both spaces. This is not a clear cross-geometry representation improvement. λ=.20 has favorable test geometry but cannot be substituted post hoc because validation selected λ=.05; it also fails downstream/risk gates.

Class-0 nearest-centroid control is not damaged by selected λ=.05 (raw .678 versus .580; normalized .713 versus .707), but this cannot overcome the failed class-4 mechanism.

## Downstream, probability, and risk results

| Condition | L1 route 4→0/1/2/3/4 | Class-4 p4 | p3+p4 | Predictive mean | Shrinkage | L1 MAE | Severe % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RPS reference | 3/6/8/3/0 | .0021 | .2800 | 1.501 | 1.345 | 2.45 | 85 |
| λ=.05 selected | 2/1/11/6/0 | .0027 | .3933 | 1.891 | 1.576 | 1.95 | 70 |
| λ=.10 | 3/7/8/2/0 | .0714 | .2242 | 1.509 | 1.278 | 2.55 | 90 |
| λ=.20 | 3/3/12/2/0 | .0025 | .2661 | 1.545 | 1.485 | 2.35 | 90 |

The selected model had zero exact L1 class-4 recovery. It improves upper-neighborhood mass, predictive mean, class-4 MAE, and severe prevalence, but inward shrinkage rises. These downstream movements cannot rescue the failed mechanism gate.

| Condition | Mode Acc/MAE/QWK/severe % | L1 Acc/MAE/QWK/severe % | L2 Acc/MAE/QWK/severe % |
| --- | --- | --- | --- |
| RPS reference | .543/.768/.499/21.3 | .525/.700/.534/18.0 | .490/.713/.528/16.8 |
| λ=.05 | .513/.768/.579/19.0 | .528/.703/.571/18.0 | .505/.693/.575/17.8 |
| λ=.10 | .523/.750/.476/21.0 | .535/.708/.524/19.0 | .520/.725/.513/19.0 |
| λ=.20 | .543/.703/.540/19.5 | .515/.695/.546/18.3 | .488/.715/.527/17.5 |

| Condition | RPS | NLL | Brier | ECE | Spearman | AUROC | AUPRC | Selective MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RPS reference | .1313 | 1.5268 | .5880 | .0941 | .433 | .617 | .244 | .473 |
| λ=.05 | .1272 | 1.3879 | .5897 | .1013 | .395 | .631 | .243 | .465 |
| λ=.10 | .1361 | 1.6406 | .6030 | .1252 | .429 | .664 | .306 | .472 |
| λ=.20 | .1292 | 1.3703 | .5823 | .0511 | .373 | .607 | .254 | .491 |

Selected λ=.05 violates predeclared tolerance for mode accuracy (−.030; allowed −.020), L1-risk Spearman (−.038; allowed −.030), and class-0 L1 MAE (+.172; allowed +.050). Its class-0 L1 accuracy/MAE/severe rate is .713/.598/23.6%, versus .741/.425/15.5% for RPS.

## Secondary fixed-head control

The predeclared fixed logit-adjusted head was evaluated only on frozen selected λ=.05 features. It raised class-4 p4 to .205 and p3+p4 to .407 but retained zero exact L1 class-4 recovery and worsened risk quality: Spearman .366, AUROC .608, AUPRC .199. It is secondary and cannot rescue the primary representation mechanism.

## Final gate and decision

The selected λ=.05 model fails the first predeclared gate: it lacks a clear class-4 representation improvement across raw and normalized analyses. It also violates class-0 and risk-quality tolerances. λ=.20 is preserved as an unselected test trade-off, not a reason to reopen method selection.

\[
\boxed{\text{NO-GO}}
\]

RG-ACR is stopped after seed-0 falsification. Do not create RG-ACR-v2 from RetinaMNIST test diagnostics. No further RetinaMNIST-test-informed objective redesign, seeds 1–4, or additional datasets are authorized by this result.

## Artifacts and limitations

Primary evidence is in outputs/retinamnist/native28/phase3_6_rg_acr_retry_1. The earlier phase3_6_rg_acr directory is preserved as an invalid partial run: a CSV schema error occurred after reference training and no scientific result was interpreted. The retry used the unchanged frozen objective and protocol.

Limitations: one seed, 20 class-4 test examples, and absence of the historical frozen RPS checkpoint requiring a matched retrained reference. These limitations do not authorize a post-hoc variant.
