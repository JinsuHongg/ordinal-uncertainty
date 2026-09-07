# Phase 3.10A — RetinaMNIST Risk-Order Preservation Objective Falsification

## Frozen protocol (written before execution)

### Scientific question and motivation

Can an established balanced frozen-head correction improve rare class-4
localization while a risk-order preservation (ROP) regularizer reduces the
resulting degradation in ordinal decision-risk quality? This follows the Phase
3.9 decomposed mechanism audit: a frozen balanced head can act on a
head-recoverable subset but has global and risk-quality costs. The candidate is
the combination, not balanced heads, cRT-style retraining, RPS, L1 Bayes risk,
or pairwise ranking individually.

### Frozen representation and split discipline

The exact canonical native-28 seed-0 RPS checkpoint and its replay-verified
512-D penultimate feature archive are reused. The backbone is never loaded for
training or differentiated through. This is a training-set-only experiment:
the 1,080 official RetinaMNIST training samples receive deterministic seed-0
stratified five-fold assignments; every sample is held out once and fitted on
four folds. Historical validation and test arrays are neither loaded nor used.
Pooled held-out predictions are the sole development evidence.

### Conditions and matched training

* A: original frozen RPS logits/probabilities, evaluated OOF by its training
  sample identities.
* B: a `512 -> 5` linear head initialized exactly from the original RPS head;
  fixed-feature CE with balanced sampling.
* C: the same head, initialization, optimizer, `lr=0.001`, `weight_decay=1e-4`,
  batch size 64, fixed 100-epoch budget, fold, and random streams as B, plus
  ROP at lambda `0.1`, `0.5`, or `1.0`.

At each step B/C use two distinct batches. `B_bal` is sampled with equal class
participation and supplies CE. `B_nat` is a shuffled empirical fitting-fold
batch and supplies ROP only. There is no class weighting, logit adjustment,
representation loss, hard-pair mining, endpoint weighting, temperature, or
checkpoint selection.

For every natural example the fixed teacher value is the exact discrete L1
Bayes risk from frozen canonical RPS probabilities. The student computes all
five differentiable action risks, selects `argmin` only on a detached tensor,
then gathers the selected action risk. For all unordered natural-batch pairs
with nonzero teacher difference, ROP is

\[
\frac{\sum w_{ij}[\max(0,-s_{ij}(r_i-r_j))]^2}
{\sum w_{ij}+10^{-12}},\quad
s_{ij}=\operatorname{sign}(r_i^{(0)}-r_j^{(0)}),\quad
w_{ij}=|r_i^{(0)}-r_j^{(0)}|.
\]

It penalizes reversals only; it contains neither absolute risk matching nor
probability distillation.

### Predeclared decision gates and tolerances

GO requires all of: (1) versus A, meaningful class-4 improvement in at least
two of MAE, shrinkage, p4, adjacent/exact routing, or severe prevalence; (2)
versus B, improvement in at least two of L1 risk/error Spearman, severe AUROC,
severe AUPRC, and mean selective MAE, with improved direct pair preservation
and no severe remaining risk degradation; (3) safety. Safety tolerances are
predeclared before execution: relative to B, no worse than `+0.03` global L1
MAE, `-0.03` QWK, `+0.10` class-0 L1 MAE, `+0.05` class-0 severe prevalence,
or `+0.05` for NLL/Brier/RPS; relative to A, no worse than `+0.06` global L1
MAE or `+0.12` class-0 L1 MAE. A GO also needs a plausible neighboring-lambda
region. Otherwise the sole decision is TRADE-OFF or NO-GO. Historical test
evaluation is prohibited regardless of result.

## Execution record

### Integrity and training diagnostics

The experiment completed locally in the existing `ordinal-uncertainty` Conda
environment (the requested `ocqr` alias was not registered). It reused
`outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/rps/seed_0/features.npz`
and `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt`.
The archive supplied only its `train_*` arrays to the script: 1,080 samples,
512 dimensions, and counts `[486,128,206,194,66]`. The source code has no
RetinaMNIST loader and does not load validation/test values. Fold assignments
were seed-0 `StratifiedKFold`: C0 `[98,97,97,97,97]`, C1 `[25,25,26,26,26]`,
C2 `[41,42,41,41,41]`, C3 `[39,38,39,39,39]`, C4 `[13,14,13,13,13]`.
Every training identity appears once held out and four times fitting.

Focused gradient/protocol tests passed. The detached student action has no
gradient while its selected L1 action risk does; gradients reach the linear
head but not frozen features or teacher probabilities. Correctly ordered pairs
have zero ROP loss; reversed pairs have positive loss; ties are excluded; all
values are finite. At epoch 100, mean ROP / pair violation / weighted violation
were `.004315/.2677/.0885` (λ=.1), `.003659/.2610/.0822` (λ=.5), and
`.003109/.2539/.0763` (λ=1.0); mean teacher-risk gap was `.3392` and ties were
zero. Thus the regularizer was active.

### Pooled OOF results

| Condition | L1 acc / MAE / QWK / severe % | NLL / Brier / RPS / ECE | Spearman / AUROC / AUPRC / selective MAE | pair / weighted pair preserved |
| --- | --- | --- | --- | --- |
| A original RPS | .518 / .696 / .604 / 19.1 | 1.141 / .554 / .122 / .035 | .467 / .674 / .269 / .393 | 1.000 / 1.000 |
| B balanced head | .507 / .744 / .612 / 19.4 | 1.236 / .603 / .136 / .072 | .471 / .700 / .354 / .427 | .727 / .912 |
| C λ=.1 | .509 / .744 / .612 / 19.4 | 1.235 / .603 / .136 / .073 | .469 / .702 / .356 / .426 | .729 / .914 |
| C λ=.5 | .511 / .741 / .613 / 19.4 | 1.230 / .601 / .135 / .077 | .472 / .703 / .351 / .422 | .737 / .922 |
| C λ=1.0 | .508 / .738 / .618 / 19.0 | 1.226 / .599 / .134 / .077 | .475 / .699 / .331 / .419 | .745 / .929 |

Mode/L2 results are retained in the machine-readable summaries. Relative to B,
λ=.5 improves Spearman, AUROC, selective MAE, both pair diagnostics, and
probability metrics, while AUPRC falls slightly. λ=1.0 improves Spearman,
selective MAE, and pair diagnostics but loses AUROC/AUPRC. λ=.1 improves
AUROC/AUPRC/selective MAE and pair preservation but lowers Spearman. This is a
smooth, modest regularization effect rather than an inactive objective.

### Rare endpoint and lower-endpoint controls

Class-4 support is 66. A/B/C(.1/.5/1.0) L1 routing `[→0,→1,→2,→3,→4]` is
`[1,1,41,23,0]`, `[0,2,15,38,11]`, `[0,3,14,38,11]`, `[0,3,14,38,11]`, and
`[0,3,14,38,11]`. Against A, every C condition retains the balanced-head
localization movement: class-4 MAE `1.136` (A `1.697`), severe prevalence
`25.8%` (A `65.2%`), p4 about `.338` (A `.116`), predictive mean about `2.72`
(A `2.20`), and inward shrinkage about `1.28` (A `1.80`). However, ROP does
not improve this recovery relative to B: B has MAE `1.121`, p4 `.338`, mean
`2.727`, and shrinkage `1.273`; all C conditions have MAE `1.136` and slightly
more shrinkage.

For class 0, A/B/C(.1/.5/1.0) L1 MAE is `.570/.733/.728/.718/.704`, severe
prevalence is `25.1/24.3/24.3/24.1/23.3%`, and accuracy is `.702/.619/.623/.630/.632`.
The C conditions partially repair B's lower-endpoint cost but still exceed the
predeclared A-relative class-0 MAE tolerance of `.12` (increases `.158/.148/.134`).

### Decision and implication

\[
\boxed{\text{TRADE-OFF}}
\]

ROP is active and gives a modest, stable improvement in portions of the
balanced-head risk trade-off, especially around λ=.5, while retaining the
balanced-head class-4 gain versus A. It does not improve rare localization over
B, and all C values fail the predeclared class-0 safety tolerance relative to
the original RPS reference. It therefore does not support an objective freeze
or select a frozen lambda. Do not create ROP-v2, evaluate historical test,
launch other datasets, or expand seeds from this result.

Artifacts: `outputs/retinamnist/phase3_10a_rop_objective_falsification/`.
