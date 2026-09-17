# Solar RPS Natural-Sampling Direction-Only Analysis

**Verdict:** **D — SOLAR RPS NATURAL DIRECTION-ONLY IS MIXED**

## Motivation and identity gate

This RPS-only follow-up asks whether natural empirical train-feature sampling
reproduces balanced direction-only C. A is the original frozen head; N changes
only classifier directions; C is the archived replacement-balanced head. This
is descriptive and does not revise frozen H1/H2a.

Only RPS seeds 1--4 were used. Their deterministic Phase-A exports have
`READY` checkpoint integrity, valid hashes, and mandatory archived-A replay
`PASS`, including exact-L1 agreement. Train/validation/readout counts are
`45,047 / 2,431 / 28,006`; readout X support is 921. The optional C replay had
small floating-point tolerance mismatches but exact-L1 agreement; C was never
refit. Solar CE is unresolved and untouched.

## Isolation and training

C's recoverable configuration is CE, AdamW (`1e-3`, zero decay), batch size
64, fixed 100 epochs, original-head initialization, and fixed original norms
and biases. N uses the same `DirectionOnlyLinear` parameterization, precision,
seed convention, feature populations, and settings. Only sampling changes:
each N epoch is one shuffled permutation of the empirical training population;
C uses replacement-balanced batches. No class weighting, resampling, logit
adjustment, or bias adjustment is used.

The archived C protocol uses a fixed terminal epoch 100. Validation features
were preserved and count-audited but not used for N selection, since validation
selection would add a difference from C. Each N fit has 70,400 steps; observed
class draws exactly equal 100 times `[12061, 11072, 16763, 4589, 562]`. Bias
error is zero and maximum fixed-norm error is `5.96e-8` or less. No backbone
training, feature regeneration, A refit, or C refit occurred in this phase.

## Endpoint results

The endpoint is exact-L1 X-class MAE on the fixed archived readout.

| Seed | A | N | C | ΔN-A | ΔN-C |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.825 | 1.464 | .827 | -.362 | .636 |
| 2 | 1.841 | 1.464 | .622 | -.378 | .841 |
| 3 | 1.582 | 1.451 | .503 | -.131 | .948 |
| 4 | 1.493 | 1.509 | .666 | .016 | .844 |

N improves 3/4 seeds: **MIXED** under the descriptive gate. Mean ΔN-A is
`-.214` (SD `.190`; t 95% CI `[-.516, .089]`). C is better than N in every
seed. Mean endpoint MAE is `1.685 / 1.472 / .654` for A/N/C.

## Global, class-wise, and mass trade-offs

| Metric | A | N | C |
| --- | ---: | ---: | ---: |
| Global MAE | .500 | .456 | .561 |
| Macro MAE | .714 | .634 | .544 |
| Global severe error | .051 | .043 | .070 |

N improves global/macro MAE and lowers severe error versus A, while C supplies
stronger endpoint and macro improvement with its known global/severe cost.
Mean endpoint-mass changes (`N-A / C-A`) across true classes 0--4 are
`-.001 / +.004`, `-.004 / +.013`, `-.005 / +.171`, `+.006 / +.360`, and
`+.022 / +.453`. N is not a broad C-like upward redistribution.

Across the four X populations, A/N/C exact-X routing is `0 / 7 / 1944` of
3,684; N mostly shifts X samples from class 2 toward class 3. X severe error
is `.673 / .462 / .170` under A/N/C.

## Interpretation and limitations

Retina N was not consistent (2/4 improvement per objective). Solar RPS has a
stronger but still mixed natural signal (3/4), whereas C is stronger in every
seed. The settings therefore do not establish generic natural-sampling
direction adaptation. Solar CE cannot be inferred from this RPS-only result.
The established balanced-C direction response remains unchanged. A bias/prior
control remains reviewer-relevant but lower priority: N does not reproduce C's
broad endpoint-mass profile, and none was run here.

This reuses the archived aligned readout rather than a new external population;
the four-seed interval is descriptive. No CE work, new seed, backbone training,
or feature extraction occurred in Phase B.

## Artifacts

- `outputs/mechanism_replication/natural_direction/solar/rps/`
- `outputs/mechanism_replication/analysis/natural_direction/solar/rps/`
- `outputs/mechanism_replication/features/solar/rps/`
