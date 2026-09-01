# Phase 3.5 — Risk-Conditioned Ordinal Representation Method-Design Audit

## Scope and decision

This is a literature-and-design audit only. No loss was implemented, no model was trained or fine-tuned, and no seed was launched. It defines one tightly predeclared falsification experiment; it does not establish novelty or effectiveness.

The unresolved mechanism is

\[
\boxed{\text{dual-component rare-extreme failure}}.
\]

Phase 3.3 established that many true class-4 samples are closer to class 2/3 than to the train-derived class-4 centroid. Phase 3.4 established that some class-4-like features are still routed centrally by the head; simple head corrections recover only a subset and introduce global/risk-quality trade-offs. Thus a representation loss targets the first component only. RPS remains the base because its ordinal decision risk aligns better with errors and severe events, even though it does not localize class 4. The design hypothesis is:

\[
\boxed{\text{decision risk may identify where representation correction is most needed}.}
\]

## Focused literature-overlap audit

Searches covered risk-aware representation learning; uncertainty-weighted contrastive learning and hard-example mining; ordinal contrastive, metric, margin, and prototype learning; and long-tailed contrastive learning. Closest verified work:

- [Rank-N-Contrast (RnC)](https://arxiv.org/abs/2210.01189): ordinal target distance controls contrastive comparisons.
- [ConOrd](https://arxiv.org/abs/2607.08109): soft affinity/disparity weights by rank gap over all batch pairs.
- [CORE](https://doi.org/10.1016/j.patcog.2024.110748): ordinal manifold alignment using prototype-constrained optimization.
- [CLOC](https://arxiv.org/abs/2504.17813): multi-margin N-pair ordinal classification.
- [PCOR-Net](https://doi.org/10.1177/08953996261433872): momentum prototypes, ordinal-aware prototype contrast, and ordinal regression.
- [Balanced Contrastive Learning](https://arxiv.org/abs/2207.09052) and [Targeted Supervised Contrastive Learning](https://arxiv.org/abs/2111.13998): long-tailed representation balancing through contrastive geometry.
- [Uncertainty-aware contrastive PU learning](https://doi.org/10.1145/3803291.3803308): predictive uncertainty weights ambiguous/border examples.

Searches explicitly included KCOC, ORL, and acronym variants. No authoritative ordinal-classification paper identifiable as KCOC was found in the searched indexes; it is not evidence of novelty. ORL is ambiguous (including unrelated object-level representation learning), so the verified works above are the reliable comparison set.

The audit therefore makes no novelty claim. A change consisting only of an uncertainty weight has **high novelty risk**. The reason to predeclare one experiment is mechanistic fit to high-risk ordinal collapse, not a claim that the mechanism is new.

## Risk signal and detachment

For softmax probabilities \(p_i\), use the established L1 Bayes risk:

\[
R_i=R^*_{L1}(x_i)=\min_{a\in\{0,\ldots,K-1\}}\sum_{k=0}^{K-1}p_{ik}|k-a|.
\]

It is aligned with the project's mandatory L1 control and captures ordinal spread, unlike entropy. Mode-centred L1 risk is a diagnostic control but depends on the mode rather than the L1-optimal action. Predictive entropy is an ablation/control only because it ignores outcome distances.

Use a stop-gradient risk weight. A differentiable \(g(R_i)\) could lower the auxiliary penalty by altering its own predicted risk instead of correcting features; it also creates unstable self-modulation with RPS. The non-detached variant is outside the seed-0 experiment. Define the single monotone bounded weight:

\[
w_i=\min\!\left(2,\frac{\operatorname{sg}(R_i)}{\operatorname{mean}_{r\in B}\operatorname{sg}(R_r)+10^{-8}}\right).
\]

The fixed cap and batch-mean normalization prevent epoch-scale drift and an uncertain tail sample from dominating. Do **not** multiply it by \(1/n_y\): generic frequency weighting already failed, and the product would turn this into rare-label up-weighting. Training counts may only support a normal class-present/stratified minibatch policy.

## Exactly three candidates

| Candidate | Failure target and mechanism | Risk / ordinal / frequency | Overlap and risks | Decision |
| --- | --- | --- | --- | --- |
| A. Risk-gated adjacent-centroid ranking (RG-ACR) | Representation collapse: make an anchor closer to its own centroid than each adjacent centroid. It does not claim to repair head bias. | Detached \(R^*_{L1}\); only \(|y-j|=1\) comparisons; no inverse-frequency multiplier. | **MEDIUM method overlap; HIGH novelty risk.** Near RnC, ConOrd, CLOC, CORE, and uncertainty-weighted metric learning. Uncertainty can identify noise; batch centroids can be noisy. | **PRIMARY CANDIDATE** |
| B. Collapse-aware adaptive adjacent margin | Adjacent centroid collapse through a training-feature running separation reference that adapts the adjacent margin. | No risk weight; ordinal adjacency and train-only separation/counts. | **MEDIUM overlap; MEDIUM-HIGH novelty risk.** Near adaptive ordinal margin methods/CLOC. Moving geometry adds feedback instability and more tuning. | **BACKUP CANDIDATE** |
| C. Risk-weighted prototype compactness | Pull high-risk samples toward an own-class prototype, optionally with ordinal prototype separation. | Detached \(R^*_{L1}\); ordinal prototype distances; no direct frequency multiplier. | **HIGH overlap; HIGH novelty risk.** Near center loss, CORE, PCOR-Net, and prototype methods. Risks prototype noise, over-compactness, and loss of uncertainty information. | **DEFER** |

Candidate A uses local relative adjacency—not all-negative SupCon, balanced SupCon, generic triplets, generic hard mining, generic center loss, or simple distance-weighted SupCon. Its decision-theoretic detached gate is the specific difference, but that difference alone is insufficient for a novelty claim.

## Candidate ranking

| Rank | Candidate | Failure fit | Novelty risk | Stability risk | Hyperparameters | Implementation cost | Seed-0 value |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 1 | RG-ACR | High for representation collapse; none for head bias | High | Medium | 1 | Low | High |
| 2 | Collapse-aware adaptive adjacent margin | Medium | Medium-high | High | 3+ | Medium | Medium |
| 3 | Risk-weighted prototype compactness | Medium | High | Medium-high | 2–3 | Medium | Low |

This is an experiment-design ordering, not a scientific novelty or performance claim.

## Primary mathematical specification: RG-ACR

Let \(h_i\in\mathbb{R}^{512}\) be the penultimate ResNet feature and \(z_i=h_i/(\|h_i\|_2+10^{-8})\). For minibatch \(B\), use a leave-one-out, L2-normalized centroid:

\[
c_{k,-i}=\frac{\sum_{r\in B:r\ne i,y_r=k}z_r}{\left\|\sum_{r\in B:r\ne i,y_r=k}z_r\right\|_2+10^{-8}}.
\]

Let \(A(y_i)=\{j:|j-y_i|=1\}\), cosine distance \(d(u,v)=1-u^\top v\), and include anchor \(i\) only when its own and every adjacent centroid are defined. With fixed margin \(m=0.05\),

\[
\ell_i=\frac{1}{|A(y_i)|}\sum_{j\in A(y_i)}[m+d(z_i,c_{y_i,-i})-d(z_i,c_{j,-i})]_+.
\]

\[
L_{\rm repr}=\frac{\sum_{i\in V_B}w_i\ell_i}{\sum_{i\in V_B}w_i+10^{-8}},
\qquad
\boxed{L=L_{\rm RPS}+\lambda L_{\rm repr}}.
\]

Here \(w_i\) is the detached bounded L1-risk weight above, \(V_B\) is the valid-anchor set, and gradients pass through anchor and centroid features but not the risk. If a required class is absent or an own-class leave-one-out centroid has no member, skip that anchor; if \(V_B\) is empty set \(L_{\rm repr}=0\) and log the event. No test feature, test centroid, or test-derived threshold enters training.

The objective is symmetric across endpoints: class 0 and 4 each compare to their sole adjacent class; interior labels compare to both neighbors. It does not presume that remote labels always require stronger repulsion. RPS supplies global probabilistic ordinal pressure; RG-ACR corrects local adjacent absorption.

There is one tunable method hyperparameter: \(\lambda\). Predeclare the validation-only grid \(\{0.05,0.10,0.20\}\), selected by minimum validation RPS. The margin, normalized features, cap, and risk normalization are fixed design conventions, not test-tuned hyperparameters.

## Predeclared seed-0 falsification and freeze rule

After separate authorization, train canonical native-28 ResNet18 on official splits with RPS base, fixed preprocessing, seed 0, and validation-only selection. Compare against matched RPS. The primary head is the jointly trained ordinary RPS probabilistic head. The sole secondary head control is the fixed Phase-3.4 prior/logit-adjusted head evaluated on learned frozen features; it is not a new head method.

Support requires: (1) improvement in at least one class-4 geometry quantity (raw/normalized 4→4 nearest-centroid routing, \(\Delta_{4,3}\), \(\Delta_{4,2}\), or class-3/class-4 separation) without material class-0 routing harm; (2) concordant class-4 improvement in \(p4\), \(p3+p4\), predictive mean/inward shrinkage, 4→3/4 L1 routing, or severe prevalence; and (3) no material degradation in Accuracy/MAE/QWK, NLL/Brier/RPS, risk-error Spearman, severe AUROC/AUPRC, or ordinal-MAE risk-coverage/selective MAE. Report mode, exact L1, exact L2, and all established class-4/class-0 metrics.

Predeclare the following test interpretation thresholds before any run: at least one geometry quantity must improve in the favorable direction by at least 10% relative to matched RPS (or, for 4→4 routing, at least two of 20 cases); at least two of the listed class-4 downstream quantities must improve favorably; class-0 L1 MAE may not rise by more than 0.05; Accuracy/QWK may not fall by more than 0.02; MAE may not rise by more than 0.03; RPS/NLL/Brier may not rise by more than 5%; and risk-error Spearman, severe AUROC/AUPRC, and mean selective MAE may not worsen by more than 0.03 absolute. These are fixed falsification safeguards, not tuning targets. If representation and routing improve with acceptable global/risk quality, declare \(\boxed{\text{METHOD FREEZE}}\) and then run seeds 1–4 before new datasets. If geometry improves but global/risk quality degrades, declare \(\boxed{\text{TRADE-OFF — REVIEW}}\), with no test-informed variants. If geometry does not improve, declare \(\boxed{\text{NO-GO}}\). RetinaMNIST is a development benchmark; this is its last major method-selection experiment.

## Remaining risks and compliance

Risk may select irreducible ambiguity/noise rather than correctable collapse; rare-class batch centroids can be noisy; improved geometry may not survive the unchanged head; and endpoint gains can worsen calibration or risk ranking. These are falsification gates, not secondary reports. Literature overlap remains material, so the audit authorizes a single test rather than a novelty claim, broad benchmarking, or a method freeze.

No code under \`src/\`, \`scripts/\`, or \`tests/\` changed. No method was implemented, trained, fine-tuned, or launched.
