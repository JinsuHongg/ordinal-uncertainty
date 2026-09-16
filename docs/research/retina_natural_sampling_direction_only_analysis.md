# RetinaMNIST Natural-Sampling Direction-Only Analysis

**Verdict:** **C — RETINA NATURAL DIRECTION-ONLY DOES NOT REPRODUCE C; BALANCED SAMPLING IS PRIMARY**

## Motivation and isolation

This follow-up tests whether the confirmatory Retina A→C response is generic
direction adaptation or depends on C's replacement-balanced sampling. It uses
only regenerated, checksummed seed-1--4 frozen features. A is the original
head; N changes only directions with natural shuffled empirical fitting-fold
samples; C is the archived replacement-balanced direction-only head. All use
the same frozen representation, original row norms and biases, CE loss, AdamW
(`1e-3`, zero weight decay), 100 epochs, and five-fold training-only OOF
assignments. C has no validation checkpoint selection, so N uses the same fixed
terminal epoch. No backbone training, feature regeneration, C refitting, Solar
access, or evaluation-driven tuning occurred.

## Endpoint results

The frozen endpoint is exact-L1 class-4 MAE. N improves over A in only 2/4
seeds for each objective, hence **NOT CONSISTENT** under the descriptive
follow-up label. C improves all four archived seeds in both settings.

| Objective | A endpoint MAE | N endpoint MAE | C endpoint MAE | mean ΔN−A (95% t CI) | N improving seeds |
|---|---:|---:|---:|---:|---:|
| CE | 1.788 | 1.773 | 1.133 | -.015 [-.806, .775] | 2/4 |
| RPS | 1.777 | 1.765 | 1.348 | -.011 [-.446, .423] | 2/4 |

N is worse than C in every seed: CE ΔN−C is `.561`, `.667`, `.712`, `.621`;
RPS ΔN−C is `.500`, `.379`, `.379`, `.409`.

## Global and redistribution trade-offs

| Objective | Global MAE A / N / C | Macro MAE A / N / C | Severe rate A / N / C |
|---|---|---|---|
| CE | .725 / .691 / .740 | .939 / .889 / .794 | .203 / .181 / .189 |
| RPS | .703 / .695 / .758 | .899 / .886 / .847 | .193 / .182 / .190 |

N has a smaller upward endpoint-mass redistribution than C for every true
class. Mean `N−A` / `C−A` endpoint-mass changes for classes 0--4 are CE:
`.002/.046`, `.003/.077`, `.010/.110`, `.023/.149`, `.041/.224`; RPS:
`.019/.067`, `.041/.129`, `.065/.186`, `.083/.230`, `.095/.258`. N creates
weak broad movement, whereas C creates a much larger class-gradient shift
concentrated near the endpoint. Detailed routing is retained in the outputs.

## Interpretation, scope, and next boundary

The outcome agrees across CE and RPS: natural direction adaptation is not
sufficient to reproduce balanced C, while balancing amplifies the response.
It independently extends the qualitative Phase 3.19 sampling conclusion from
one frozen RPS backbone to four confirmatory Retina backbones per objective.
It remains RetinaMNIST-specific and descriptive; it does not alter frozen
H1/H2a or establish unique causality of sampling.

The Retina claim should therefore say **balanced direction-only response**, not
generic natural-sampling direction adaptation. A bias/prior-only control remains
a lower-priority reviewer concern because C's broad endpoint-mass redistribution
is not reproduced by N; it was not run here. Solar remains separate cluster
work and was not touched.

## Artifacts

Canonical tables: `outputs/mechanism_replication/analysis/natural_direction/retina/`.
N outputs: `outputs/mechanism_replication/natural_direction/retina/`.
