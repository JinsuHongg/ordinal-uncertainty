# Phase 3.12 — Evidence Consolidation and Candidate Disposition

## Scope and evidence ledger

This is an analysis-only disposition of completed Phase 3.10B–3.11 evidence.
No model was trained, no split was evaluated, and no RetinaMNIST test artifact
was opened. The evidence consists of five-fold, training-only OOF head audits
and falsifications on frozen 512-D seed-0 RPS features, followed by the single
predeclared full-training historical-validation gate. It is RetinaMNIST
mechanism evidence, not a cross-dataset or final-method result.

| Phase | Supported observation | Limitation / status |
|---|---|---|
| 3.10B parameter audit | Balanced recovery is weight-driven: C4 direction cosine `.314`, norm `.575→3.285`, and C4 margins rise through feature projection; bias shifts are only about `.003–.008`. | The same global change lowers the C0 margin by `3.908`; 84 C0 cases are damaged. Bias swaps are inert. **Replicated mechanism audit.** |
| 3.10C direction-only | C4 MAE improves A→C `1.697→1.348`; exact recovery is `2`; C0 MAE `.691` and global L1 MAE `.721` improve B's `.733/.744`. | Only `1/11` B-exact recoveries remain. Fixed original scale is too restrictive. **Replicated OOF causal evidence.** |
| 3.10D controlled scale | α `.50–.75` is a non-isolated OOF region. At α=.50: C4 MAE `1.258`, exact `8`, C0 MAE `.698`, global MAE `.731`; B is `1.121/11/.733/.744`. | It is a fixed, predeclared RetinaMNIST mechanism probe, not a universal scale rule. **Replicated OOF causal evidence.** |
| 3.10E controlled scale + ROP | E improves OOF C4 MAE `1.258→1.167`, Spearman `.460→.466`, selective MAE `.427→.422`, pair preservation `.732→.747`, weighted `.917→.928`. | AUROC/AUPRC fall `.704/.342→.692/.303`; NLL/Brier/RPS/ECE worsen slightly. **OOF-only support.** |
| 3.11 one-shot validation | A→E C4 MAE `1.500→1.167`, p4 `.122→.328`, mean `2.268→2.701`, shrinkage `1.732→1.299`: outward localization replicates. | With only six C4 examples, exact routing is descriptive. B→E C0/global MAE reverse `.630→.722` and `.642→.667`; D→E Spearman/selective MAE reverse `.450→.438` and `.359→.366`; pair `.803→.798`. **Mixed validation evidence.** |

Thus rare-end outward movement is **replicated** across OOF and validation;
the controlled-scale safety advantage and ROP's downstream risk effect are **not
replicated** on validation; opposite-endpoint/global safety and cross-dataset
transport remain **unresolved**.

## Component disposition

| Component | Decision | Evidence basis |
|---|---|---|
| Bias correction | **STOP** | Phase 3.10B found negligible bias changes. Original weights plus balanced biases reproduce original behavior; balanced weights plus original biases reproduce B. Bias/logit adjustment is not an active mechanism here. |
| Direction adaptation | **RETAIN** | C4 margins, alignment, and localization improve through direction rotation. Direction-only training improves C4 MAE over A and partly reduces B's cost; validation preserves outward C4 movement. |
| Scale amplification | **RETAIN** | Removing scale inflation loses most exact recovery; direction-only retains only `1/11` B-exact recoveries. Some increased scale is necessary for stronger recovery within the tested causal controls. |
| Controlled scale | **RETAIN AS MECHANISTIC FINDING** | OOF α=.50–.75 shows a useful non-isolated localization/safety region between fixed-original and unrestricted scale. Validation does not establish its safety as a final rule. |
| ROP | **RETAIN AS DIAGNOSTIC / SECONDARY IDEA** | Its fixed OOF interaction improves direct ordering, Spearman, and selective MAE, but all three D→E directions fail to reproduce on validation. It is not robust enough as a proposed final method component. |
| Combined α=.50, λ=1.0 candidate | **HOLD — NOT FREEZE-READY** | Phase 3.11's predeclared outcome is mixed: localization reproduces, while safety and complementarity do not. No α/λ change is permitted. |

The evidence supports the following qualified RetinaMNIST decomposition:

\[
\text{rare-end localization benefit}
\approx
\text{direction adaptation} + \text{controlled scale},
\]

while ROP robustness is unconfirmed and opposite-endpoint/global collateral
damage remains an unresolved safety question. This is a mechanism statement,
not an assertion that any particular head is a validated method.

## Novelty and research framing

Generic classifier-scale control is not a standalone novelty claim: it overlaps
with decoupled long-tail classifiers, \(\tau\)-normalization, LWS, MaxNorm /
weight balancing, generalized normalization/scaling, and ordinal-imbalance work
such as KCOC. The surviving contribution is the diagnosis of rare ordinal
endpoint localization under imbalance and its opposite-endpoint/global
decision-risk trade-off—not a renamed generic scale-control algorithm.

The established cross-dataset phenomenon remains rare upper-extreme inward
localization bias. The direction/scale explanation is presently
RetinaMNIST-specific. It requires a frozen confirmation protocol rather than a
method grid.

## Project-level decision

\[
\boxed{\text{A — CROSS-DATASET MECHANISM CONFIRMATION JUSTIFIED}}
\]

The next scientific question, if separately executed under a frozen UTKFace
protocol, is:

> Does the direction/scale localization mechanism observed on RetinaMNIST
> reproduce on UTKFace under a frozen, predeclared protocol?

The minimal comparison is original RPS head, balanced head, direction-only
head, and controlled-scale head. ROP is excluded unless separately justified;
there is no automatic ROP, alpha/lambda grid, adaptive-scale design, test
evaluation, seed expansion, or new method authorization.
