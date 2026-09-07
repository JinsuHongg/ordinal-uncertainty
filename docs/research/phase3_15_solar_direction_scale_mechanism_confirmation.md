# Phase 3.15 — Solar Direction/Scale Mechanism Confirmation

## Question and frozen protocol

Phase 3.15 asked whether the frozen RPS Solar head exhibits the same
classifier-direction/scale decomposition observed on RetinaMNIST and UTKFace.
It is a mechanism audit, not a search for a Solar classifier.

The exact Phase 3.8 RPS seed-0 checkpoint and Phase 3.9 frozen feature cache
were reused: 512-dimensional penultimate features, channel order
`[hmi_m, aia1600, aia131]`, source indices `[8, 7, 1]`, and the existing
aligned identities. The train/validation/test feature counts were
`45,047/2,431/28,006`; class-X test support was `921`. IDs were disjoint across
all splits, all feature/logit values were finite, and original-test-logit
replay error was `1.34e-5`.

B/C/D were fit only on the archived aligned training features. The validation
archive was loaded only for provenance/integrity, with no selection or early
stopping. The already-used Phase 3.8 test set was explicitly authorized as one
archived confirmatory readout after the entire protocol was frozen; it was
never used for fitting or selection.

All heads began from the original RPS head and used balanced CE, AdamW
`lr=.001`, batch `64`, seed `0`, and exactly `100` fixed epochs. B trained a
full linear head with weight decay `1e-4`. C trained directions only with
original row norms and biases fixed. D trained directions only with original
biases and predeclared fixed row norms

\[
s_k^D=.5\|w_k^A\|+.5\|w_k^B\|.
\]

Direction-only weight decay was `0`, ROP was absent, and no backbone,
representation, channels, alpha values, losses, CE counterpart, or additional
seeds were used. Job `4334303` completed on the `qGPU24` partition
(`acidsgcn011`) in `37:01`; failed/cancelled setup retries `4333857`, `4333858`,
and `4334300` are preserved and were not scientifically interpreted.

## Mechanical integrity

C fixed the original norms exactly (maximum error `0`); D's maximum fixed-norm
error was `4.77e-7`. The balanced head substantially amplified all row norms:
class-X `.594 -> 17.266`; D set its X norm to `8.930`; C retained `.594`.
Classifier directions moved substantially: X cosine with A was `.143` for B,
`.405` for C, and `.164` for D. D remained close to B in direction (X cosine
`.900`). Balanced bias changes were material for classes 0/3 (`-.219/+ .228`)
but negligible for X (`+.006`); this is diagnostic, not a bias intervention.

## Rare-X localization (L1 decisions)

| Condition | X routing 0/1/2/3/4 | Exact X | X MAE | severe | mean / median pX | predictive mean | shrinkage |
|---|---:|---:|---:|---:|---:|---:|---:|
| A original RPS | 0/22/137/762/0 | 0/921 | 1.197 | 17.3% | .033/.032 | 2.772 | 1.228 |
| B balanced | 0/23/333/231/334 | 334/921 | 1.049 | 38.7% | .385/.270 | 2.941 | 1.059 |
| C direction-only | 0/22/153/250/496 | 496/921 | .675 | 19.0% | .513/.545 | 3.227 | .773 |
| D controlled scale `.50` | 0/22/240/397/262 | 262/921 | 1.024 | 28.4% | .326/.215 | 2.952 | 1.048 |

B establishes head actionability: it moves exact X recovery from zero to 334
and raises mean pX from `.033` to `.385`, although its severe burden and
probability/global quality worsen. C is the clearest Solar mechanism result:
with original scale and bias fixed, it recovers 496 exact X decisions, lowers X
MAE to `.675`, raises mean pX to `.513`, and reduces shrinkage to `.773`.
Thus direction adaptation alone is not merely partial here; it is stronger than
B on the primary X endpoint.

D does not add the expected extra outward movement: compared with C it loses
234 exact X recoveries, raises X MAE by `.349`, lowers pX by `.186`, and raises
shrinkage by `.275`. It remains better than A but does not retain more B
recovery than C already has.

## Lower endpoint and global collateral behavior

| Condition | C0 MAE / severe | C0 predictive mean | Global L1 accuracy / MAE / QWK / severe | NLL / Brier / RPS / ECE | Spearman / AUROC / AUPRC / selective MAE |
|---|---:|---:|---:|---:|---:|
| A | .082 / 1.9% | .172 | .585 / .452 / .808 / 3.3% | 1.012 / .576 / .081 / .169 | .243 / .628 / .047 / .325 |
| B | .095 / 1.6% | .148 | .506 / .585 / .740 / 8.6% | 1.431 / .727 / .108 / .263 | .239 / .619 / .123 / .446 |
| C | .086 / 1.9% | .198 | .512 / .552 / .786 / 6.1% | 1.110 / .643 / .095 / .170 | .148 / .495 / .055 / .449 |
| D | .074 / 1.5% | .141 | .534 / .538 / .764 / 6.8% | 1.181 / .653 / .097 / .197 | .220 / .602 / .091 / .409 |

Solar's lower endpoint does not exhibit the Retina balanced-head pattern:
B slightly worsens C0 MAE, C is near baseline, and D improves C0 MAE. But no
adapted head improves global L1 MAE or the original RPS probability/risk
metrics. These are collateral effects, not failure of the primary direction
mechanism.

## Decision and cross-dataset interpretation

\[
\boxed{\text{PARTIAL CONFIRMATION}}
\]

Solar strongly confirms head actionability and independently confirms the
importance of classifier-direction adaptation. It does **not** confirm the
prior RetinaMNIST/UTKFace ordering in which controlled intermediate scale adds
to direction-only localization: at Solar's frozen `.50` scale, D is worse than
C on every primary rare-X localization quantity. Therefore Solar strengthens
the cross-dataset direction claim but narrows the scale statement:

> Across RetinaMNIST, UTKFace, and Solar, rare upper-end localization is
> responsive to classifier-direction adaptation. Additional fixed scale can
> strengthen outward movement on RetinaMNIST and UTKFace, but is not a
> universal improvement over direction-only adaptation.

This result does not propose a method and does not authorize tuning alpha,
ROP, bias correction, representation retraining, new datasets, or seed
expansion. Any next step requires separate authorization.

Artifacts: `outputs/solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24/`.
