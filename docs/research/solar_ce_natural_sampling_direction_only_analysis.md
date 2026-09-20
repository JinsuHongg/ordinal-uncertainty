# Solar CE natural-sampling direction-only analysis

**Verdict:** **C — SOLAR CE N FOLLOW-UP COMPLETE; NOT CONSISTENT**

## Motivation and status

This auxiliary follow-up asks whether natural empirical-sampling direction-only
adaptation reproduces the large rare-end response of archived balanced C on
Solar CE.  It is not part of the original preregistered confirmatory H1/H2a
record; those historical analyses remain separately preserved.

Seeds 1, 3, and 4 use existing validated CE frozen features.  The historical
seed-2 feature archive is excluded solely because it failed the prespecified
numerical feature-identity gate required for this follow-up; the one coherent,
pre-declared replacement seed-2 A/C/N unit is used instead.  No tolerance was
relaxed.  No backbone was rerun for seeds 1, 3, or 4, and N was fit exactly
once for each remaining seed.

## Frozen protocol and validation

The frozen protocol is
`docs/research/solar_ce_natural_sampling_remaining_seeds_protocol.md`.
All original seeds passed the Phase-A feature gate (`READY`, archived-A replay
`PASS`) before fitting.  Their SHA256-validated float32, 512-D archives have
45,047 / 2,431 / 28,006 train/validation/readout rows, exact A/C readout
ID/label alignment, and 921 true-X readout rows.  Direct A-head replays from
the frozen eval features were within the frozen `2e-5` logit and `2e-6`
probability thresholds, with zero exact-L1 differences.

N starts from each original A head and optimizes only unit class directions
under CE.  Original per-class norms and biases remain fixed; N and C use AdamW
(`1e-3`, zero decay), batch 64, 100 fixed epochs, and original-head
initialization.  N's only changed operation is one shuffled empirical,
without-replacement training-feature pass per epoch; C uses replacement
class-balanced batches.  N has no validation selection because archived C has
none.  Each fit used 70,400 steps and exactly 100 times the empirical class
counts.  The maximum norm error was `5.96e-8` and bias error was zero.

## Per-seed endpoint results

Endpoint performance is true-X exact-discrete-L1 MAE on the shared 28,006-row
readout.

| Unit | A | N | C | N-A | C-A | N-C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.052 | 1.488 | .422 | +.435 | -.630 | +1.065 |
| replacement 2 | 1.667 | 1.355 | .573 | -.312 | -1.093 | +.782 |
| 3 | 1.590 | 1.377 | .518 | -.213 | -1.072 | +.859 |
| 4 | 1.413 | 1.485 | .637 | +.073 | -.775 | +.848 |

N improves over A in 2/4 units.  Its mean N-A endpoint MAE is `-.004` (sample
SD `.335`; descriptive t 95% CI `[-.538, .529]`), hence the frozen auxiliary
label is **natural-direction response not consistent**.  C is better than N
in every unit.

## Global and class-wise trade-offs

| Metric | A | N | C |
| --- | ---: | ---: | ---: |
| Endpoint MAE | 1.430 | 1.426 | .538 |
| Global MAE | .492 | .436 | .587 |
| Macro MAE | .654 | .605 | .534 |
| Global severe error | .046 | .037 | .084 |

Across the four true-X populations, exact-X routing is `110 / 129 / 2,234`
for A/N/C.  N's endpoint correction is weak and heterogeneous: it improves
replacement seed 2 and seed 3, but worsens seed 1 and seed 4.  C produces a
substantially stronger endpoint response in every unit, alongside its known
global/severe cost.

## Endpoint-mass and routing redistribution

Mean endpoint-probability shifts (N-A / C-A) by true class 0--4 are:
`-.0004 / +.0045`, `-.0025 / +.0153`, `-.0008 / +.2012`,
`+.0217 / +.4215`, and `+.0380 / +.4953`.  N therefore creates a small shift
concentrated at the upper classes, not C's broad and much larger upward
redistribution.  The detailed class-wise endpoint mass and exact-L1 routing
are retained in the canonical CSVs.

## Cross-setting synthesis

| Setting | N improves A | N weaker than C | Descriptive reading |
| --- | --- | --- | --- |
| Retina CE | 2/4 | 4/4 | natural N does not reproduce C |
| Retina RPS | 2/4 | 4/4 | natural N does not reproduce C |
| Solar RPS | 3/4 | 4/4 | partial natural signal; C remains stronger |
| Solar CE | 2/4 | 4/4 | natural response not consistent |

Natural-sampling direction-only adaptation can improve rare-end localization
in some units, but it does not generally reproduce the large and consistent
response obtained with balanced direction-constrained adaptation.  This does
not establish that balanced sampling alone causes the effect: the fixed
direction-only parameterization governs how the sampling change is expressed
by the classifier head.  It also does not change the frozen confirmatory
H1/H2a conclusions.

## Limitations

The replacement seed-2 unit is a transparently documented new coherent
replication unit, not the historical seed-2 model.  The four-unit descriptive
interval is not a preregistered hypothesis test, and it uses the archived
aligned readout rather than a new external population.

## Artifacts

- Protocol: `docs/research/solar_ce_natural_sampling_remaining_seeds_protocol.md`
- N outputs: `outputs/mechanism_replication/natural_direction/solar/ce/`
- Analysis: `outputs/mechanism_replication/analysis/natural_direction/solar/ce/`
- Replacement seed-2 provenance:
  `docs/research/solar_ce_seed2_replacement_run.md`
