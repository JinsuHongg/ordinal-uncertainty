# Phase 3.10E — Controlled Scale × Risk-Order Preservation Interaction

## Question and frozen protocol

Phase 3.10A established that the exact risk-order preservation (ROP) hinge is
active, but that it was tested on an overly aggressive full balanced head.
Phase 3.10D established a useful controlled-scale region, selecting the
predeclared α=.50 point as the strongest recovery/safety comparator. This
single interaction test asks whether fixed ROP can improve ordinal risk quality
there without undoing controlled-scale localization or endpoint safety.

All results are pooled training-only five-fold OOF predictions over the 1,080
canonical RetinaMNIST training samples. They reuse the Phase 3.10A assignments,
frozen `(1080,512)` canonical seed-0 RPS features and teacher probabilities,
original RPS bias, restored B fold norms, and Phase 3.10D balanced sampler.
No historical validation/test sample, backbone gradient, new data, seed, alpha,
or lambda search was used.

D has fixed (s_k=.5s_k^A+.5s_k^B), original bias, and trainable classifier
direction. E is exactly D plus the Phase 3.10A zero-margin, teacher-risk
weighted squared-hinge ROP loss on a separate empirical-distribution natural
batch, with fixed \(\lambda=1.0\). CE uses a separate balanced batch. AdamW
uses LR `.001`, batch size 64, 100 fixed epochs, and zero direction weight
decay: radial decay is unidentifiable after weight normalization. Only the five
direction vectors are trainable.

Every final effective norm equaled its target to at most `4.77e-8`; directions
were initialized exactly at the original directions (maximum normalized
direction error `0`). The detached discrete L1 action and the established ROP
pair construction/loss were reused directly from Phase 3.10A.

## Pooled OOF results

| Condition | L1 accuracy / MAE / QWK / severe % | NLL / Brier / RPS / ECE |
|---|---|---|
| A original RPS | .518 / .696 / .604 / 19.1 | 1.141 / .554 / .122 / .035 |
| B full balanced | .507 / .744 / .612 / 19.4 | 1.236 / .603 / .136 / .072 |
| D α=.50 | .503 / .731 / .619 / 18.9 | 1.221 / .599 / .133 / .060 |
| E α=.50 + ROP | .506 / .729 / .623 / 18.2 | 1.225 / .600 / .134 / .069 |

Relative to D, E improves global L1 MAE by `.003`, QWK by `.004`, and severe
prevalence by `.006`. Its probability costs are modest but real: NLL `+.004`,
Brier `+.001`, RPS `+.001`, ECE `+.010`; none regresses to the full-balanced B
level.

| Condition | L1-risk Spearman | severe AUROC | severe AUPRC | mean selective MAE | pair / weighted pair preserved |
|---|---:|---:|---:|---:|---:|
| D α=.50 | .460 | .704 | .342 | .427 | .732 / .917 |
| E α=.50 + ROP | .466 | .692 | .303 | .422 | .747 / .928 |

ROP improves two downstream risk measures (Spearman and selective MAE) and
both direct order diagnostics, while AUROC and AUPRC decline. Thus the result
is not claimed from the internal pair statistic alone. Natural-batch final-state
activity was nonzero across folds: mean ROP loss `.0016–.0033`, violation
fraction `.215–.265`, weighted violation fraction `.058–.092`, and no teacher
risk ties. The pooled violation fraction falls from `.268/.083` (unweighted /
weighted) for D to `.253/.072` for E.

## Localization, safety, and mechanism

| Condition | C4 routing 0/1/2/3/4 | C4 MAE / exact / severe % | C4 p4 / mean / shrinkage | C0 routing 0/1/2/3/4 | C0 MAE / severe % |
|---|---|---|---|---|---|
| D α=.50 | 0/3/19/36/8 | 1.258 / 8 / 33.3 | .302 / 2.640 / 1.360 | 296/81/71/36/2 | .698 / 22.4 |
| E α=.50 + ROP | 0/3/14/40/9 | 1.167 / 9 / 25.8 | .325 / 2.689 / 1.311 | 301/75/65/44/1 | .702 / 22.6 |

E retains and modestly improves D's localization: one additional exact class-4
decision, C4 MAE improves `.091`, and severe C4 prevalence falls `.076`. C0
MAE changes only `+.004` and severe prevalence `+.002`; it remains clearly
safer than B (`.733` C0 MAE). Relative to B, E retains 9/11 exact recoveries
and 33/37 inward-improved class-4 cases; it restores 16/84 B-damaged class-0
samples.

The mean key margins shift from D to E as follows: `z4-z3: -.075 -> .049`,
`z4-z2: .223 -> .350`, and `z0-z1: 1.150 -> 1.104`. E's final directions stay
very close to D (mean foldwise cosine by class 0–4: `.994, .995, .996, .993,
.995`) while rotating substantially from A (`.768, .594, .647, .553, .466`).
ROP therefore acts as a mild directional regularizer around the controlled-scale
solution, not as a scale or bias change.

Five-fold means ± standard deviations for D versus E are global L1 MAE
`.731±.076` versus `.729±.049`, C4 MAE `1.260±.187` versus `1.166±.054`, and
C0 MAE `.697±.111` versus `.702±.106`. These descriptive fold values are not
treated as independent-seed significance tests.

## Decision

\[
\boxed{\text{GO — COMPLEMENTARY MECHANISMS SUPPORTED}}
\]

At this one predeclared interaction point, ROP adds pair preservation,
Spearman, and selective-MAE benefit while preserving—and modestly improving—the
controlled-scale class-4 result with no material class-0/global degradation.
Its AUROC/AUPRC and probability-quality costs remain a limitation. This is
training-only mechanism evidence, not a final method, freeze, test evaluation,
or authorization for alpha/lambda tuning, adaptive scale, ROP-v2, seeds, or
other datasets. ROP remains scientifically justified only for this fixed
controlled-scale interaction; any further work requires separate authorization.

Artifacts: `outputs/retinamnist/phase3_10e_controlled_scale_rop_interaction/`.
