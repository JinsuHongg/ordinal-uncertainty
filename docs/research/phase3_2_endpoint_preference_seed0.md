# Phase 3.2 — True-Endpoint-Preference Correction, Seed 0

## Motivation, scope, and guardrail

Candidate 1 showed that `-log(p3+p4)` can move true class-4 L1 decisions away
from central class 2, but it credits classes 3 and 4 equally. Candidate 1b is
the smallest fixed refinement: it gives class 4 twice the auxiliary credit of
class 3. Only seed 0 was run; Candidates 2/3 and seeds 1--4 were not run.

This is the final RetinaMNIST seed-0 objective refinement for the Candidate-1
branch. Objective, `rho`, endpoint weights, lambda grid, checkpoint selection,
and evaluation were fixed before Candidate-1b test results. Do not create a
Candidate 1c from these test diagnostics or alter this objective after them.

## Objective and implementation

\[
\ell=\ell_{\mathrm{RPS}}+
\lambda\mathbf1\{y\in\{0,K-1\}\}w_y
\left[-\log\left(p_y+\rho\sum_{k:|k-y|=1}p_k\right)\right],
\qquad \rho=0.5.
\]

Hence the endpoint terms are `-log(p0 + 0.5 p1)` and `-log(p4 + 0.5 p3)`;
interior labels have zero auxiliary loss. `rho=0.5` is predeclared, not tuned.
Endpoint weights retain Candidate 1's training-count-only formulation,

\[
w_y=\sqrt{\min(n_0,n_{K-1})/n_y}.
\]

For train counts `[486, 128, 206, 194, 66]`, `w0=0.3686` and `w4=1.0000`.
This is relative endpoint balancing / majority-endpoint attenuation.

Candidate 1 and 1b share the existing trainer through a narrow objective
configuration. Candidate 1b is restricted in code to `rho=0.5` and
`lambda in {0.1, 0.3}`. Tests cover zero interior loss, mirrored endpoint
symmetry, lambda-zero RPS equivalence, far-mass penalty, finite gradients, and
the critical contrast: equal `p3+p4` yields equal Candidate-1 loss but Candidate
1b favors higher p4.

## Protocol

Canonical native-28 RetinaMNIST, official split, small-image ResNet18, batch
64, AdamW (`1e-3`, `1e-4` weight decay), 20 epochs, and seed 0 were unchanged.
Each checkpoint was selected only by minimum validation RPS. The lambda=.1,
rho=.5, two-epoch smoke run was finite, finalized all artifacts, and selected
epoch 2 with validation RPS `.177006`; it has no scientific interpretation.

| lambda | rho | selected epoch | validation RPS |
| ---: | ---: | ---: | ---: |
| .1 | .5 | 7 | .114723 |
| .3 | .5 | 7 | .110239 |

The frozen CE/RPS/WCE/SLACE prediction files are absent from this ignored-output
checkout. Repository-wide search and current utilities cannot recover frozen
RPS median p4, median p3+p4, or class-4 L1 Bayes risk without RPS retraining;
RPS was not retrained. Documented Phase 3.1 seed-0 values remain the reference.

## Global and decision results

| Method | Mode accuracy / MAE / QWK / severe % | NLL / Brier / RPS / ECE | L1 accuracy / MAE / QWK / severe % | L2 accuracy / MAE / QWK / severe % |
| --- | --- | --- | --- | --- |
| Frozen RPS | .538 / .695 / .576 / 19.25 | 1.213 / .566 / .122 / .062 | .548 / .660 / .599 / 18.25 | unavailable |
| C1, lambda .1 | .510 / .788 / .555 / 18.50 | 1.399 / .592 / .126 / .071 | .525 / .683 / .594 / 16.25 | .453 / .738 / .548 / 16.75 |
| C1, lambda .3 | .510 / .800 / .558 / 19.50 | 1.699 / .664 / .137 / .128 | .505 / .780 / .570 / 18.50 | .488 / .720 / .563 / 17.50 |
| C1b, lambda .1 | .480 / .850 / .492 / 21.00 | 1.569 / .615 / .138 / .085 | .483 / .798 / .498 / 22.25 | .460 / .783 / .488 / 21.00 |
| C1b, lambda .3 | .503 / .845 / .506 / 22.75 | 1.364 / .598 / .128 / .084 | .515 / .718 / .564 / 16.75 | .470 / .715 / .565 / 16.00 |

## Primary class-4 L1 routing and geometry

| Method | 4->0/1/2/3/4 | MAE | Severe % | Exact 4->4 |
| --- | --- | ---: | ---: | ---: |
| Frozen RPS | 1/2/15/2/0 | 2.10 | 90 | 0 |
| C1, lambda .1 | 2/4/4/10/0 | 1.90 | 50 | 0 |
| C1, lambda .3 | 2/4/0/14/0 | 1.70 | 30 | 0 |
| C1b, lambda .1 | 2/4/8/6/0 | 2.10 | 70 | 0 |
| C1b, lambda .3 | 3/3/3/11/0 | 1.90 | 45 | 0 |

| Method | Mean / median p3 | Mean / median p4 | Mean / median p3+p4 | Predictive mean | Shrinkage | Mean L1 risk |
| --- | --- | --- | --- | ---: | ---: | ---: |
| C1, lambda .1 | .3251 / .3448 | .1263 / .1419 | .4514 / .4916 | 1.9722 | 1.4986 | .9540 |
| C1, lambda .3 | .4020 / .4489 | .1206 / .1421 | .5226 / .5910 | 2.0254 | 1.4220 | .8165 |
| C1b, lambda .1 | .2627 / .2761 | .1161 / .1127 | .3787 / .3963 | 1.8385 | 1.5052 | .9044 |
| C1b, lambda .3 | .3465 / .3957 | .1176 / .1079 | .4641 / .5280 | 1.9469 | 1.5238 | .9651 |

Candidate 1b fails the primary mechanism test: p4 is lower than the same-lambda
Candidate 1 result at both settings, near-endpoint mass is lower, no exact class
4 decision appears, and adjacent `4->3` recovery weakens (6 versus 10 at .1;
11 versus 14 at .3).

## Class-0 control and risk quality

| Method | 0->0/1/2/3/4 | Accuracy / MAE / severe % | Mean p0 / p1 / p0+p1 | Mean / shrinkage / L1 risk |
| --- | --- | --- | --- | --- |
| C1, .1 | 125/12/29/8/0 | .718 / .540 / 21.26 | .700 / .079 / .780 | .708 / .655 / .475 |
| C1, .3 | 122/15/7/30/0 | .701 / .684 / 21.26 | .689 / .119 / .808 | .714 / .596 / .403 |
| C1b, .1 | 113/17/33/11/0 | .649 / .667 / 25.29 | .682 / .075 / .757 | .750 / .668 / .449 |
| C1b, .3 | 129/18/14/13/0 | .741 / .489 / 15.52 | .745 / .068 / .813 | .611 / .552 / .407 |

| Method | L1 Spearman | Severe AUROC | Severe AUPRC | Mean selective MAE | Mode->L1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen RPS | .399 | .668 | .283 | .409 | unavailable |
| C1, .1 | .360 | .641 | .255 | .464 | 41.00 |
| C1, .3 | .428 | .646 | .261 | .498 | 6.75 |
| C1b, .1 | .416 | .674 | .374 | .531 | 27.25 |
| C1b, .3 | .400 | .606 | .203 | .467 | 31.50 |

Candidate 1b lambda .1 has stronger ranking/detection values but the worst
selective MAE and poor global/class-4 behavior. Lambda .3 retains Spearman near
RPS but loses AUROC/AUPRC and remains worse than RPS in selective MAE. RPS-like
risk geometry is not jointly retained with true-endpoint improvement.

## Decision

\[
\boxed{\text{CANDIDATE 1 BRANCH — NO-GO}}
\]

No lambda is a method-freeze or multi-seed candidate. Candidate 1b did not
raise p4 over Candidate 1, yielded no `4->4`, and weakened Candidate 1's
adjacent recovery. Do not design Candidate 1c from RetinaMNIST test results;
future work must use a different reviewed mechanism family and separate
authorization.

Artifacts are isolated under
`outputs/retinamnist/native28/phase3_2_endpoint_preference/`. Verification:
`pytest -q` 25 passed; `python -m compileall src scripts` passed;
`git diff --check` passed.
