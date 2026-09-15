# RetinaMNIST Confirmatory H1/H2 Analysis

**Status:** Complete — Retina-only confirmatory analysis of seeds 1--4
**Date:** 2026-09-14
**Scope:** Frozen H1 and H2a analyses only. This is not a cross-dataset claim,
does not establish imbalance causality, and does not alter the frozen protocol.

## Frozen analysis specification

Inputs are the eight completed, integrity-verified A/C condition artifacts:

- `outputs/mechanism_replication/ac/retina/ce/seed_{1,2,3,4}/`
- `outputs/mechanism_replication/ac/retina/rps/seed_{1,2,3,4}/`

The only independent replication units are backbone seeds 1--4. Each seed's
1,080-row artifact contains five-fold training-only OOF A/C predictions; folds
are not treated as independent replications. H1 uses exact-L1 class-4 MAE and
defines the primary paired effect as `C MAE - A MAE`, where negative is
beneficial.

For H2a, each seed contributes its 66 true class-4 held-out OOF rows. The
outcome is final C inward shrinkage `Y_C = 4 - predictive_mean_C`; the original
A-state control is `Y_A = 4 - predictive_mean_A`; endpoint-specific geometry is
`m_adj = d3 - d4`; and generic geometry is `g = d_(2) - d_(1)`. Predictors are
standardized using the current seed's class-4 rows and sample SD (`ddof=1`);
the outcome is not standardized. The full-sample M1 coefficient has HC3 robust
standard errors. Literal LOOCV re-estimates every predictor's standardization
on the 65-row training portion of each held-out class-4 sample.

## H1 — rare-end L1 localization

| Objective | Seed | A MAE | C MAE | Delta MAE | A exact | C exact | A shrinkage | C shrinkage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 1 | 1.2273 | 1.1667 | -0.0606 | 0 | 7 | 1.5302 | 1.3055 |
| CE | 2 | 2.2424 | 1.1818 | -1.0606 | 0 | 5 | 2.1010 | 1.3346 |
| CE | 3 | 1.7121 | 1.3182 | -0.3939 | 0 | 2 | 1.9271 | 1.4904 |
| CE | 4 | 1.9697 | 0.8636 | -1.1061 | 0 | 24 | 1.9641 | 1.1427 |
| RPS | 1 | 1.6212 | 1.4394 | -0.1818 | 0 | 0 | 1.9241 | 1.5361 |
| RPS | 2 | 1.6667 | 1.3788 | -0.2879 | 0 | 0 | 2.0738 | 1.4255 |
| RPS | 3 | 1.9394 | 1.2576 | -0.6818 | 0 | 5 | 1.9991 | 1.3641 |
| RPS | 4 | 1.8788 | 1.3182 | -0.5606 | 0 | 1 | 2.0879 | 1.4283 |

| Objective | Negative seeds | Mean Delta | SD | 95% t CI | H1 verdict |
| --- | ---: | ---: | ---: | --- | --- |
| CE | 4 / 4 | -0.6553 | 0.5130 | [-1.4716, 0.1610] | REPLICATED |
| RPS | 4 / 4 | -0.4280 | 0.2326 | [-0.7981, -0.0580] | REPLICATED |

The binary H1 verdict uses only the frozen 4-of-4 sign gate; the confidence
interval is descriptive rather than an additional gate.

## H2a — endpoint-specific geometry and final C localization

M0 contains standardized `Y_A` and `g`; M1 additionally contains standardized
`m_adj`. `beta_ord` is the M1 coefficient for `m_adj`.

| Objective | Seed | beta_ord | HC3 SE | HC3 95% CI | MSE M0 | MSE M1 | Delta_LOO |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| CE | 1 | -0.0692 | 0.0703 | [-0.2070, 0.0685] | 0.02361 | 0.02379 | -0.00018 |
| CE | 2 | -0.0701 | 0.0859 | [-0.2384, 0.0982] | 0.07403 | 0.07549 | -0.00147 |
| CE | 3 | -0.0726 | 0.0352 | [-0.1416, -0.0036] | 0.02520 | 0.02475 | 0.00044 |
| CE | 4 | -0.1809 | 0.0651 | [-0.3084, -0.0534] | 0.07453 | 0.06654 | 0.00799 |
| RPS | 1 | 0.1430 | 0.0252 | [0.0935, 0.1924] | 0.02226 | 0.01574 | 0.00652 |
| RPS | 2 | -0.0813 | 0.0346 | [-0.1492, -0.0135] | 0.01895 | 0.01812 | 0.00083 |
| RPS | 3 | -0.0377 | 0.0696 | [-0.1741, 0.0987] | 0.02157 | 0.02757 | -0.00600 |
| RPS | 4 | 0.0253 | 0.0295 | [-0.0325, 0.0831] | 0.01052 | 0.01027 | 0.00025 |

| Objective | beta_ord < 0 count | Delta_LOO > 0 count | Mean Delta_LOO | H2a verdict |
| --- | ---: | ---: | ---: | --- |
| CE | 4 / 4 | 2 / 4 | 0.00170 | PARTIAL |
| RPS | 2 / 4 | 3 / 4 | 0.00040 | NOT SUPPORTED |

CE satisfies the coefficient-sign portion in all four seeds but does not meet
the required at-least-three-positive-LOOCV-seed condition, so its frozen H2a
verdict is PARTIAL. RPS has beneficial `beta_ord` in only two of four seeds,
so its frozen H2a verdict is NOT SUPPORTED. These are setting-specific
RetinaMNIST results, not a general ordinal mechanism conclusion.

## Secondary H2b and diagnostics

H2b is descriptive only and cannot change H2a. Per-seed mean changes and
Pearson correlations with `m_adj` are saved in `h2b_secondary.csv`.

| Objective | Seed | Mean Delta_mu | Mean Delta_p_end | r(m_adj, Delta_mu) | r(m_adj, Delta_p_end) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE | 1 | 0.2248 | 0.1309 | 0.1015 | 0.7927 |
| CE | 2 | 0.7664 | 0.2536 | -0.2750 | 0.5045 |
| CE | 3 | 0.4367 | 0.1897 | -0.2296 | 0.7181 |
| CE | 4 | 0.8215 | 0.3199 | 0.0563 | 0.7468 |
| RPS | 1 | 0.3880 | 0.2698 | -0.2903 | 0.6749 |
| RPS | 2 | 0.6484 | 0.2820 | 0.1506 | 0.8339 |
| RPS | 3 | 0.6350 | 0.2000 | 0.5665 | 0.8411 |
| RPS | 4 | 0.6596 | 0.2819 | 0.7439 | 0.8668 |

The required m_adj diagnostic correlations with original endpoint probability,
original `z4-z3` margin, generic `g`, and original shrinkage `Y_A` are saved
in `diagnostics.csv`. They are collinearity/construct diagnostics, not separate
replication evidence and not covariates added to M1.

## Output artifacts and limitations

Machine-readable outputs are in:

- `outputs/mechanism_replication/analysis/retina/ce/`
- `outputs/mechanism_replication/analysis/retina/rps/`
- `outputs/mechanism_replication/analysis/retina/retina_confirmatory_summary.json`

The analysis uses four independently trained backbones per objective and only
the RetinaMNIST training-only OOF evaluation population. It does not establish
universality, isolate imbalance as a cause, or provide a cross-dataset claim.
