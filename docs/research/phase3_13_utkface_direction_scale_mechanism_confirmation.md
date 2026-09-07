# Phase 3.13 — UTKFace Direction/Scale Mechanism Confirmation

## Scientific question and frozen protocol

This confirmatory phase asked whether the RetinaMNIST direction/scale mechanism
transports to UTKFace: does balanced direction adaptation plus some scale
amplification improve rare upper-end localization, and does controlled scale
avoid the unrestricted head's lower-endpoint/global cost?

The archived Phase 3.7A manifest was reused from
`/home/jhong90/github_proj/ordinal-aware-conformal/data/split_assignments/conference_v0_3/utkface/manifest.jsonl`.
The canonical seed-0 RPS checkpoint was
`outputs/utkface/phase3_7a_failure_replication/rps/seed_0/best_checkpoint.pt`.
Its backbone was frozen and used on `cuda:0` in evaluation/inference mode to
extract 512-D features. The fitting split contained 14,224 samples with counts
`[2756, 7128, 2726, 1210, 404]`; the archived validation split contained 2,371
samples with counts `[459, 1188, 455, 202, 67]`. IDs were unique within each
split and disjoint across them. Features were finite. The test split was not
constructed or loaded.

Four conditions were fixed before evaluation:

- **A:** untouched original RPS head.
- **B:** full linear head, original-head initialization, balanced CE.
- **C:** direction-only balanced CE with original per-class norms and biases.
- **D:** direction-only balanced CE with fixed
  `s_D = .5 s_A + .5 s_B` and original biases.

B/C/D used AdamW, learning rate `.001`, batch size `64`, 100 fixed epochs, and
the same balanced sampler. B used weight decay `1e-4`; C/D used zero decay on
normalized raw direction parameters. No ROP, class weights, alternate alpha,
checkpoint selection, backbone training, or post-result variant was used.

## Infrastructure and integrity

The recovery host exposed an NVIDIA GeForce GTX 1660 Ti. PyTorch
`2.13.0+cu130` reported CUDA available with one device, and extraction ran
explicitly on `cuda:0` with `num_workers=0`. Separate atomic caches were saved
as `frozen_train_features.npz` and `frozen_val_features.npz`. Both record the
checkpoint and extraction device. Fitted B/C/D states and complete fixed-epoch
histories were saved before validation metrics were computed.

Maximum fixed-norm errors were `5.96e-8` for C and `2.38e-7` for D. Thus both
constrained heads satisfied the scale contract to the required tolerance.

## Global, probability, and risk results

All decision metrics below use the exact L1 Bayes decision.

| Condition | Accuracy | MAE | QWK | Severe | NLL | Brier | RPS | ECE | Spearman | AUROC | AUPRC | Selective MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | .7453 | .2691 | .8191 | .0131 | .6583 | .3563 | .04984 | .0365 | .3822 | .7908 | .0712 | .1381 |
| B | .7440 | .2737 | .8362 | .0160 | .6360 | .3552 | .05035 | .0327 | .3987 | .8119 | .1144 | .1279 |
| C | .7035 | .3138 | .8209 | .0156 | .6899 | .3908 | .05544 | .0180 | .4001 | .7706 | .0779 | .1588 |
| D | .7086 | .3096 | .8235 | .0164 | .6838 | .3881 | .05489 | .0332 | .3956 | .7752 | .1018 | .1565 |

B slightly worsened global L1 MAE relative to A but improved QWK, probability
NLL/Brier, risk Spearman, AUROC/AUPRC, and selective MAE. C and D incurred a
larger global accuracy/MAE and probability-quality cost than B. D was slightly
better than C globally, but it did not supply a RetinaMNIST-like joint global
safety advantage over unrestricted B.

## Rare upper endpoint and lower-endpoint control

| Condition | C4 routing 0/1/2/3/4 | C4 exact | C4 MAE | C4 severe | Mean p4 | Predictive mean | Shrinkage | Mean L1 risk |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | `[0,2,5,25,35]` | 35 | .6119 | .1045 | .4988 | 3.2454 | .7546 | .4334 |
| B | `[0,2,2,25,38]` | 38 | .5224 | .0597 | .5389 | 3.4080 | .5920 | .2624 |
| C | `[0,2,2,24,39]` | 39 | .5075 | .0597 | .5766 | 3.4181 | .5819 | .3241 |
| D | `[0,2,2,23,40]` | 40 | .4925 | .0597 | .5914 | 3.4497 | .5503 | .2821 |

All adapted heads moved class 4 outward on multiple diagnostics. Direction-only
C slightly exceeded B, and intermediate-scale D was strongest, although the
increment from C to D was modest. This confirms that direction adaptation is
useful on UTKFace and is compatible with scale amplification; it does not show
that unrestricted B is the maximum-localization endpoint.

| Condition | C0 routing 0/1/2/3/4 | C0 MAE | C0 severe | Mean p0 | Predictive mean | Mean L1 risk |
|---|---|---:|---:|---:|---:|---:|
| A | `[313,143,2,1,0]` | .3268 | .0065 | .6588 | .3955 | .1790 |
| B | `[385,67,5,2,0]` | .1808 | .0153 | .8366 | .2162 | .0987 |
| C | `[387,59,11,2,0]` | .1895 | .0283 | .8302 | .2488 | .1371 |
| D | `[395,54,8,2,0]` | .1656 | .0218 | .8529 | .2099 | .1078 |

Unlike RetinaMNIST, unrestricted B did not damage class-0 MAE: every adapted
condition improved it substantially relative to A, and D had the lowest C0
MAE. Severe-error prevalence moved differently and was higher for B/C/D than
A. The central opposite-endpoint damage pattern therefore did not replicate.

## Parameter decomposition

| Class | Norm A | Norm B | Norm D | B bias shift | cos(B,A) | cos(C,A) | cos(D,A) | cos(C,B) | cos(D,B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .659 | 3.110 | 1.884 | -.061 | .296 | .797 | .492 | .443 | .889 |
| 1 | .605 | 2.673 | 1.639 | -.017 | .312 | .724 | .496 | .550 | .892 |
| 2 | .581 | 2.245 | 1.413 | +.020 | .373 | .661 | .500 | .521 | .865 |
| 3 | .602 | 3.621 | 2.112 | -.053 | .350 | .669 | .406 | .577 | .946 |
| 4 | .688 | 6.634 | 3.661 | +.108 | .250 | .703 | .270 | .391 | .940 |

Large balanced-head norm inflation and direction rotation reproduce clearly.
D learns directions close to B while holding the predeclared intermediate
norms. The B bias shifts are larger than the negligible RetinaMNIST shifts,
especially for classes 0 and 4, so the bias diagnostic does not transport;
this audit does not establish that those shifts are causal.

## RetinaMNIST comparison and decision

| Predeclared pattern | UTKFace result |
|---|---|
| B improves C4 localization over A | **Replicated.** C4 MAE `.612→.522`, exact `35→38`, shrinkage `.755→.592`. |
| C retains part of B's benefit while reducing collateral damage | **Localization replicated; safety pattern not replicated.** C improves C4 beyond B, while no B-like C0 MAE damage exists to remove and C is globally worse. |
| D recovers more C4 than C while safer than B | **Partly replicated.** D is best for C4 and C0 MAE, but its global MAE/probability quality remain materially worse than B. |
| Negligible balanced-head bias movement | **Not replicated.** C0/C4 shifts are `-.061/+.108`. |

There is no fold-level variability estimate because the established confirmatory
protocol fits once on the archived training split and evaluates once on the
archived validation split. This is a frozen seed-0 mechanism confirmation, not
a statistical hypothesis test or a universal-superiority claim.

\[
\boxed{\text{PARTIAL CONFIRMATION}}
\]

The rare-end localization component transports: balanced direction adaptation
is useful, large scale changes accompany the effect, and fixed intermediate
scale D gives the strongest class-4 localization. The RetinaMNIST
opposite-endpoint/global safety ordering does not transport cleanly. Thus the
project-level claim should retain a cross-dataset direction/scale localization
signal while explicitly treating the safety trade-off and bias contribution as
dataset-dependent. No UTKFace-specific candidate, further alpha, ROP, backbone
training, test evaluation, extra seed, Solar run, or method redesign is
authorized by this result.
