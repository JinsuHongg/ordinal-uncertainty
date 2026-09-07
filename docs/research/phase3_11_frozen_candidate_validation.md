# Phase 3.11 — Frozen Candidate One-Shot Validation

## Frozen candidate and split discipline

This one-shot gate evaluates the already-frozen Phase 3.10E candidate only:
controlled scale \(\alpha=.50\), original RPS bias, trainable classifier
directions, balanced CE, and the exact detached-action Phase 3.10A ROP hinge
on a separate natural-distribution batch with \(\lambda=1.0\). No alpha,
lambda, epoch, objective, threshold, checkpoint, or implementation choice was
selected from validation.

The canonical seed-0 RPS representation remained frozen. B, D, and E were
each fitted on all 1,080 canonical training samples (`[486,128,206,194,66]`)
from the original RPS head using AdamW, LR `.001`, 100 fixed epochs, batch 64.
B uses balanced CE and weight decay `.0001`; D/E train normalized direction
parameters only, retain original biases, and use zero raw-direction weight
decay. D/E norms are fixed to `.5 ||w_A|| + .5 ||w_B||`, where B is the newly
fitted full-training balanced head, not an OOF average.

The fitting command only materialized `train_*` arrays. It saved B, D, and E
before validation was loaded; `fitting_complete.json` records
`validation_loaded: false`, `test_loaded: false`, all required state paths, and
the frozen α/λ. The separate evaluation command refuses to load validation
unless those states exist. It materializes only `val_*` arrays; no test array
is indexed or materialized. Final D/E norm error is `5.96e-8`, and E's final
training ROP loss/violation fraction is `.00173/.2305`. There is no
validation-selected checkpoint.

## One-shot historical-validation results

Validation has 120 samples with counts `[54,12,28,20,6]`; in particular,
class-4 support is only six and exact routing is descriptive rather than a
hard gate.

| Condition | L1 accuracy / MAE / QWK / severe % | NLL / Brier / RPS / ECE |
|---|---|---|
| A original RPS | .583 / .583 / .662 / 15.0 | 1.025 / .513 / .111 / .063 |
| B full balanced | .525 / .642 / .670 / 12.5 | 1.155 / .563 / .117 / .121 |
| D controlled scale | .533 / .650 / .662 / 12.5 | 1.180 / .578 / .119 / .143 |
| E frozen candidate | .542 / .667 / .637 / 15.0 | 1.169 / .568 / .119 / .123 |

| Condition | risk/error Spearman | severe AUROC / AUPRC | selective MAE | pair / weighted pair preserved |
|---|---:|---:|---:|---:|
| D | .450 | .751 / .238 | .359 | .803 / .959 |
| E | .438 | .745 / .261 | .366 | .798 / .959 |

## Endpoint diagnostics

| Condition | C4 L1 routing 0/1/2/3/4 | C4 MAE / p4 / mean / shrinkage / severe % | C0 L1 routing 0/1/2/3/4 | C0 MAE / severe % |
|---|---|---|---|---|
| A | 0/0/3/3/0 | 1.500 / .122 / 2.268 / 1.732 / 50.0 | 36/3/13/2/0 | .648 / 27.8 |
| B | 0/0/2/4/0 | 1.333 / .305 / 2.664 / 1.336 / 33.3 | 35/9/6/3/1 | .630 / 18.5 |
| D | 0/0/2/2/2 | 1.000 / .362 / 2.734 / 1.266 / 33.3 | 35/9/5/3/2 | .667 / 18.5 |
| E | 0/0/2/3/1 | 1.167 / .328 / 2.701 / 1.299 / 33.3 | 35/6/8/3/2 | .722 / 24.1 |

E improves class-4 MAE, `p4`, combined upper mass (`p3+p4=.601` versus
`.467`), predictive mean, shrinkage, and severe prevalence relative to A. It
does not simply reproduce unrestricted B: D has the strongest outward movement
and two exact class-4 decisions, while E has one. However, E does not preserve
the OOF safety point on this split: its C0 MAE is worse than B by `.093` and its
global L1 MAE worse by `.025`. This endpoint finding has 54 class-0 examples,
but remains a single historical split rather than a new seed estimate.

## OOF-to-validation direction of effect

| Predeclared comparison | OOF direction | Validation direction | Assessment |
|---|---|---|---|
| A → E C4 MAE | improves (1.697→1.167) | improves (1.500→1.167) | same direction |
| A → E C4 mean / shrinkage | outward / lower shrinkage | outward / lower shrinkage | same direction |
| B → E C0 MAE | improves (.733→.702) | worsens (.630→.722) | opposite |
| B → E global MAE | improves (.744→.729) | worsens (.642→.667) | opposite |
| D → E Spearman | improves (.460→.466) | worsens (.450→.438) | opposite |
| D → E selective MAE | improves (.427→.422) | worsens (.359→.366) | opposite |

The validation result therefore confirms the broad rare-upper localization
effect, but does not reproduce the OOF controlled-scale safety or complementary
ROP risk-order effect. AUROC changes slightly downward and AUPRC upward from D
to E, consistent with the previously noted divergence among risk diagnostics.

## Decision

\[
\boxed{\text{MIXED — HOLD FROZEN, NO REDESIGN}}
\]

The class-4 localization signal is supported across multiple continuous and
ordinal diagnostics, but the full OOF localization/safety/risk interaction does
not replicate cleanly on the untouched validation split. The candidate remains
unchanged: no α/λ tuning, alternative epoch, objective revision, ROP-v2, or
test evaluation is authorized by this result. **METHOD FREEZE is not
authorized.** A separately authorized decision is required before any further
evaluation.

Artifacts: `outputs/retinamnist/phase3_11_frozen_candidate_validation/`.
