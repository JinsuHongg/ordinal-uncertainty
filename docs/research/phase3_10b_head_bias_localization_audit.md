# Phase 3.10B — RetinaMNIST Head-Bias Localization Audit

## Question, provenance, and frozen protocol

This is a mechanism audit of the completed Phase 3.10A **TRADE-OFF**, not a
new method. It asks which parameter/logit changes in the balanced frozen head
recover class 4 and damage class 0. It reuses only Phase 3.10A's training-only
1,080-sample OOF structure, seed-0 fold assignments, saved frozen 512-D RPS
features, canonical RPS checkpoint, and B-condition held-out predictions.
Historical validation/test data were not loaded.

The Phase 3.10A archive omitted B fold states. This was a provenance gap,
because parameter-level analysis was impossible without them. The exact fixed
100-epoch CPU protocol was replayed only to restore the missing linear states;
each reconstructed state reproduced its saved held-out logits with maximum
absolute error `0.0`. No backbone, feature, label split, or scientific
comparison was changed or rerun.

For `z_k=w_k^T h+b_k`, the audit reports feature term `u_k=w_k^T h`, bias,
logit, normalized feature/classifier alignment, and exact logit-margin
decomposition. Original weights are the canonical checkpoint head; B weights
are the fold-specific balanced heads. Diagnostic swaps use original weights /
balanced biases and balanced weights / original biases only; they are not
candidate methods.

## Parameters, priors, and classifier geometry

Fitting-fold priors remain approximately `[.450,.119,.191,.180,.061]` (log
priors `.798` through `-2.795`). Rebalancing moves biases only weakly: C0
`-.0084 -> -.0156` (Δ `-.0072`), C3 `-.0378 -> -.0357` (Δ `.0021`), and C4
`.0046 -> .0079` (Δ `.0034`). This direction is superficially consistent with
rebalancing C0/C4, but is far too small to establish a causal prior mechanism.

| Class | Original bias | Balanced bias | Δ bias | Original norm | Balanced norm | Δ norm | Direction cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -.008 | -.016 ± .004 | -.007 | .533 | .891 ± .029 | +.357 | .682 ± .034 |
| 1 | -.045 | -.053 ± .003 | -.008 | .548 | 1.173 ± .063 | +.625 | .581 ± .031 |
| 2 | .029 | .035 ± .003 | +.006 | .541 | 1.115 ± .035 | +.574 | .551 ± .024 |
| 3 | -.038 | -.036 ± .002 | +.002 | .538 | 1.656 ± .326 | +1.118 | .440 ± .067 |
| 4 | .005 | .008 ± .003 | +.003 | .575 | 3.285 ± .301 | +2.710 | .314 ± .038 |

The corresponding mean angular rotations are about 47° (C0), 64° (C3), and
72° (C4). Thus neither stable direction nor a bias-only explanation is
credible; class-4 norm growth is also very large.

## Alignment and margin decomposition

For true class 4, original versus balanced mean alignment to `(w2,w3,w4)` is
`(.099,.106,-.064)` versus `(.000,.021,.011)`. Mean feature terms are
`(.634,.682,-.428)` versus `(-.001,.423,.457)`; mean logits are
`(.663,.645,-.423)` versus `(.034,.387,.465)`. For true class 0, alignment to
`(w0,w1)` changes from `(.245,-.121)` to `(.049,.006)`, and feature terms from
`(3.243,-1.873)` to `(1.455,.247)`.

| Condition | Mean / median z4-z3 | Positive | Mean z4-z2 | Positive | Mean / median z0-z1 | Positive |
|---|---:|---:|---:|---:|---:|---:|
| Original | -1.068 / -1.032 | 0.000 | -1.087 | 0.000 | 5.153 / 3.897 | .963 |
| Balanced | .078 / .197 | .667 | .431 | .697 | 1.246 / 1.636 | .695 |

The original-to-balanced shifts are almost wholly projection/weight-driven:
for `z4-z3`, total `+1.146 = +1.144` feature `+ .001` bias; for `z4-z2`,
`+1.517 = +1.520` feature `-.003` bias; for `z0-z1`, `-3.908 = -3.908`
feature `+.001` bias. The same global weight changes therefore lift class-4
margins while collapsing the class-0 margin.

## Recovery, damage, and counterfactuals

There are 11 exact class-4 recoveries, 37 inward-improved class-4 samples, and
84 damaged class-0 samples. Their mean changes in the relevant margin are:

| Subset | N | Δ feature | Δ bias | Δ total |
|---|---:|---:|---:|---:|
| Exact C4 recovery | 11 | +1.735 | +.003 | +1.738 |
| C4 inward improved | 37 | +1.255 | +.002 | +1.257 |
| C0 damaged | 84 | -1.544 | +.000 | -1.544 |

Diagnostic swaps corroborate this. Original weights plus balanced biases is
nearly unchanged from original (global MAE `.697`, C4 MAE `1.697`, zero exact
C4, C0 MAE `.570`). Balanced weights plus original biases is nearly the full
balanced outcome (global MAE `.744`, C4 MAE `1.121`, 11 exact C4, C0 MAE
`.730`). Hence bias changes are not the operative source of either recovery or
damage.

Because norm changes were large, a non-tuned normalization diagnostic was run:
each balanced head retained its directions/biases but all row norms were set to
that fold's original balanced mean norm. It yielded global MAE `.697`, QWK
`.630`, C4 MAE `1.227` with one exact C4, and C0 MAE `.669`. Scale inflation
therefore amplifies both the benefit and the harm, but removing it also removes
most exact class-4 recovery; direction changes remain material.

ROP λ=1.0 is not analyzed secondarily because Phase 3.10A did not save ROP
head states and that comparison was optional. Reconstructing those states is
not needed to answer the primary B-versus-original mechanism question.

## Decision and implication

\[
\boxed{\text{MIXED PARAMETER MECHANISM}}
\]

The balanced-head trade-off is dominated by large classifier direction changes
and weight-scale inflation, not by biases. A bias-only future correction is not
justified. A norm-aware or direction-aware component could be relevant, but the
normalization diagnostic shows neither component alone is established as a
solution; any future intervention would need a separately authorized mixed
design and frozen evaluation plan. No new method, loss, backbone training,
validation/test evaluation, or dataset was implemented or run.

Artifacts: `outputs/retinamnist/phase3_10b_head_bias_localization_audit/`.
