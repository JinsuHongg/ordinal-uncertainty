# Phase 3.8 — Solar Rare-Extreme Shrinkage Confirmation

## Decision

\[
\boxed{\text{STRONG CONFIRMATION}}
\]

The solar-flare seed-0 matched-baseline gate clearly reproduces the central
cross-dataset phenomenon: the rare upper endpoint (X, class 4) is strongly
inward-localized, materially harder than the lower endpoint, and is not
recovered by exact L1 or L2 decisions. This decision concerns the
cross-dataset rare-upper-extreme shrinkage hypothesis, not universal RPS
superiority. No new method, channel variant, seed expansion, or follow-up audit
is authorized by this result.

## Question and frozen protocol

Primary question: does rare upper-extreme inward localization bias observed on
RetinaMNIST and UTKFace also occur in ordinal solar-flare classification?
Secondary question: does RPS improve risk quality or X localization relative to
matched CE? RPS was allowed to win, tie, or lose.

The canonical source was `/scratch/users/jhong36/data/surya-bench-224.zarr`,
with frozen manifests under `/scratch/users/jhong36/data`. The target was
derived only from `max_goes_class`: FQ/A→0, B→1, C→2, M→3, X→4. Binary
`label_max` was not used. Input was exactly `[hmi_m, aia1600, aia131]`, selected
from authoritative source indices `[8, 7, 1]` in that non-numerically-sorted
order, yielding `3×224×224` tensors. No channel search, 13-channel experiment,
or fallback was run.

Timestamp alignment reused the audited explicit nanosecond conversion:
`Zarr time ns - 8 hours = manifest timestamp ns`. The setup is **future-period
test evaluation with temporally overlapping train/validation years**: test is
approximately 2020–2024; train/validation are primarily 2010–2019 and overlap
in year. It is neither fully chronological nor i.i.d./temporally exchangeable.

## Data integrity and preprocessing

The original / aligned class counts (C0–C4) were:

| Split | Original | Aligned | Aligned C0–C4 | Aligned X |
|---|---:|---:|---|---:|
| Train | 74,760 | 45,047 | 12,061 / 11,072 / 16,763 / 4,589 / 562 | 562 |
| Validation | 3,672 | 2,431 | 513 / 779 / 873 / 242 / 24 | 24 |
| Test | 43,848 | 28,006 | 5,284 / 4,474 / 9,935 / 7,392 / 921 | 921 |

The original manifest counts matched the frozen contract. Alignment missingness
was 39.7%, 33.8%, and 36.1% for train/validation/test. It varied by test year
from 32.1% to 39.2% and was not strongly class-dependent: test class-specific
missingness was 34.0%–38.5%. The predeclared integrity gate passed; X support
was usable in every split. This retained-subset/missingness caveat remains a
limitation.

The valid historical train-only artifact
`outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json`
was integrity-checked and reused without modification. It applies
`sign(x) * log1p(abs(x))` then train-only channel means
`[-0.0021253, 2.6567077, 1.2993773]` and standard deviations
`[1.0077459, 1.6469010, 0.7595000]`. Training only used independent horizontal
and vertical flips.

## Model, objectives, and execution

CE and RPS used the same unpretrained torchvision ResNet18 (3 input channels,
5 outputs), initialization policy, data subset, preprocessing, augmentation,
AdamW optimizer, batch size 16, learning rate `5e-5`, weight decay `0.01`, seed
0, maximum 300 epochs, and early-stopping patience 3. CE used categorical CE
and selected minimum validation CE; RPS used the verified normalized
differentiable five-class RPS and selected minimum validation RPS. Only loss and
corresponding validation criterion differed.

All substantive work used SLURM: prepare `4276828` completed (1m01s), GPU smoke
`4276830` completed (1m23s), CE `4276834` completed (6h08m50s), and RPS
`4276835` completed (6h17m30s). The first pending smoke submission `4276829`
was cancelled before execution after a wrapper-path defect; it wrote no
experimental artifact. Each job used 1 GPU, 8 CPUs, 64 GB RAM, and a 24-hour
limit. Smoke verified GPU forward/backward, finite CE/RPS losses, labels,
channel shape, and output contracts. No direct login-node training occurred.

Both selected epoch 1: CE validation CE 1.10225; RPS validation RPS 0.09127.
Subsequent validation loss rose through the patience window, so this is
validation-only early stopping rather than test selection.

## Global test results

| Method | Decision | Accuracy | MAE | QWK | Severe % |
|---|---|---:|---:|---:|---:|
| CE | Mode | 0.5565 | 0.4811 | 0.8050 | 3.346 |
| CE | L1 | 0.5767 | 0.4565 | 0.8139 | 2.946 |
| CE | L2 | 0.5972 | 0.4312 | 0.8228 | 2.532 |
| RPS | Mode | 0.5686 | 0.4719 | 0.8000 | 3.699 |
| RPS | L1 | 0.5849 | 0.4517 | 0.8077 | 3.335 |
| RPS | L2 | 0.6047 | 0.4270 | 0.8164 | 2.853 |

Probability metrics (CE / RPS): NLL 1.06443 / 1.01153; Brier 0.60573 /
0.57627; RPS 0.08408 / 0.08069; ECE 0.20313 / 0.16852. RPS had slightly
better global probability and L1/L2 MAE metrics, but did not improve the
primary X-class localization outcome.

Using L1 decision and L1 Bayes risk, CE / RPS risk-quality values were:
Spearman 0.2538 / 0.2428; severe AUROC 0.6887 / 0.6279; severe AUPRC 0.04933 /
0.04703; mean ordinal-MAE selective risk 0.32336 / 0.32543 (lower is better).
Thus RPS did not improve risk/error association, severe detection, or selective
ordinal prediction in this seed-0 solar run.

## Class-wise L1 audit

| Class | n | CE acc / MAE / severe % / mean p(true) / mean risk | RPS acc / MAE / severe % / mean p(true) / mean risk |
|---|---:|---|---|
| 0 | 5,284 | .9245 / .1041 / 2.23 / .8714 / .1169 | .9417 / .0816 / 1.89 / .8799 / .1204 |
| 1 | 4,474 | .5103 / .5040 / 1.43 / .4055 / .3860 | .4240 / .5811 / .51 / .3549 / .3924 |
| 2 | 9,935 | .2533 / .7738 / 2.72 / .2671 / .3534 | .4064 / .6303 / 3.67 / .3715 / .3898 |
| 3 | 7,392 | .8749 / .1710 / 3.92 / .7687 / .1949 | .7400 / .3052 / 3.88 / .6508 / .2651 |
| 4 (X) | 921 | .0000 / 1.1151 / 9.01 / .0401 / .1630 | .0000 / 1.1965 / 17.26 / .0334 / .1981 |

Class-4 risk is higher than lower-endpoint risk in both methods (.1630 vs
.1169 CE; .1981 vs .1204 RPS), although central classes can also have high
risk. The endpoint comparison is decisive: X MAE is about 10.7× / 14.7× the
class-0 MAE, and X severe burden is about 4.0× / 9.1× class 0 for CE/RPS.

## Primary X-class and lower-endpoint audit

For true X, full routing `[4→0,4→1,4→2,4→3,4→4]` was:

| Method | Mode | L1 | L2 |
|---|---|---|---|
| CE | [0, 29, 57, 835, 0] | [0, 23, 60, 838, 0] | [0, 17, 61, 843, 0] |
| RPS | [0, 23, 146, 752, 0] | [0, 22, 137, 762, 0] | [0, 21, 130, 770, 0] |

Neither model made one exact X decision under any required decision rule. L1
and L2 move a small number of errors but leave every X sample inward-routed,
mostly to class 3. This directly establishes that decision correction is
insufficient.

| Method | X p4 mean / median | X p3 mean / median | X p3+p4 mean / median | Predictive mean | Inward shrinkage | L1 risk mean / median | X MAE / severe % |
|---|---|---|---|---:|---:|---|---|
| CE | .0401 / .0353 | .8117 / .9072 | .8518 / .9558 | 2.8639 | 1.1361 | .1630 / .0945 | 1.1151 / 9.01 |
| RPS | .0334 / .0316 | .7369 / .8481 | .7703 / .8908 | 2.7719 | 1.2281 | .1981 / .1501 | 1.1965 / 17.26 |

The lower-endpoint control is sharply different. Full class-0 routing (mode / L1 / L2) was CE `[4914,258,75,37,0]` / `[4885,281,85,33,0]` / `[4804,356,99,25,0]` and RPS `[5012,177,71,24,0]` / `[4976,208,77,23,0]` / `[4869,304,90,21,0]`. L1 accuracy was .9245 / .9417, MAE .1041 / .0816, severe 2.23% / 1.89%, mean p0 .8714 / .8799, mean p1 .0900 / .0812, p0+p1 .9614 / .9610, predictive mean (inward displacement) .1794 / .1715, and mean L1 risk .1169 / .1204. Thus the rare upper endpoint is much harder and pulled more than one ordinal class inward, while the lower endpoint is localized close to zero.

## Cross-dataset assessment

| Finding | RetinaMNIST | UTKFace | Solar |
|---|---|---|---|
| Upper extreme has elevated risk | REPLICATED | REPLICATED | REPLICATED |
| Upper extreme is inward-shrunk | REPLICATED | REPLICATED | REPLICATED |
| Upper extreme harder than lower endpoint | REPLICATED | REPLICATED | REPLICATED |
| Exact upper-extreme localization poor | REPLICATED | PARTIALLY REPLICATED | REPLICATED |
| L1/L2 correction insufficient | REPLICATED | PARTIALLY REPLICATED | REPLICATED |
| RPS improves risk/error association | REPLICATED | NOT REPLICATED | NOT REPLICATED |
| RPS improves severe-error detection | REPLICATED | PARTIALLY REPLICATED | NOT REPLICATED |
| RPS improves ordinal selective prediction | REPLICATED | NOT REPLICATED | NOT REPLICATED |

Solar therefore strengthens the cross-dataset localization-bias hypothesis but
also reinforces the evidence against a universal RPS risk-quality claim.

## Limitations and implications

This is one seed and one fixed three-channel representation, not a channel or
architecture comparison. Alignment retains only a subset and the future test
period differs from the overlapping train/validation years. The 921 aligned X
test examples support a strong descriptive seed-0 gate, not an unrestricted
universal claim. Neither test outcome was used to tune a method.

The third independent dataset now gives strong confirmation that rare upper
extremes can remain probability-shrunk and inward-routed under ordinal
imbalance even when CE/RPS global metrics differ. It does not identify a
mechanism, establish RPS superiority, or authorize a new loss, representation
or head audit, multi-seed run, additional dataset, or channel variant. Any next
method or confirmation decision requires separate authorization.

Artifacts: `outputs/solar/phase3_8_shrinkage_confirmation/`.
