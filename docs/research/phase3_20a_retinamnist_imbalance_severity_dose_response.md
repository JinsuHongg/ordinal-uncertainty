# Phase 3.20A — RetinaMNIST Controlled Imbalance-Severity Dose Response

## Decision

\[
\boxed{\text{B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT}}
\]

Controlled reductions in class-4 training support changed the learned class-4
probability and head margin consistently, but did not produce a clear,
seed-consistent monotonic worsening in every primary localization outcome.
The project must therefore retain the bounded wording that inward localization
is observed in studied imbalanced settings; this experiment does not justify a
stronger general controlled-causality claim.

## Frozen protocol and integrity

The predeclared 5 × 5 grid trained 25 unpretrained native-28 CE ResNet18
models, one for every seed `0–4` and `N4 ∈ {66,50,33,16,8}`. The only changed
training input was class-4 support. Classes 0–3 stayed at `[486,128,206,194]`;
all runs used random-horizontal-flip training augmentation, normalized native
28×28 RGB input, AdamW (lr `.001`, weight decay `1e-4`), batch size `64`, 20
epochs, and minimum validation NLL checkpoint selection. Test access occurred
only in the predeclared final evaluation of every completed cell.

Every run saved its exact selected class-4 IDs. For each seed the 8/16/33/50
subsets are nested within the 66-example set; all non-class-4 IDs are identical
across severities. All 25 checkpoints, histories, predictions/logits,
probabilities, train-derived centroids, and test features are present. There
are no duplicate test IDs and manifests declare no validation/test member in
training.

## Seed-aggregate results

Values are mean ± SD over the five seeds; decisions use exact discrete L1 Bayes
actions. Raw nearest-centroid geometry is train-subset-derived separately for
each severity, including its severity-specific class-4 centroid.

| N4 | C4 MAE | Shrinkage | Predictive mean | Mean p4 | Severe C4 | Mean z4-z3 | Raw nearest-4 | C0 MAE | Global L1 MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 2.330 ± .239 | 2.356 ± .127 | 1.644 ± .127 | .092 ± .044 | .830 ± .189 | -1.230 ± .229 | .500 ± .177 | .478 ± .029 | .721 ± .035 |
| 50 | 2.280 ± .104 | 2.323 ± .096 | 1.677 ± .096 | .062 ± .019 | .880 ± .125 | -1.615 ± .443 | .370 ± .091 | .507 ± .044 | .706 ± .032 |
| 33 | 2.380 ± .130 | 2.398 ± .131 | 1.602 ± .131 | .045 ± .011 | .890 ± .082 | -1.878 ± .156 | .360 ± .102 | .500 ± .034 | .704 ± .033 |
| 16 | 2.410 ± .164 | 2.472 ± .108 | 1.528 ± .108 | .025 ± .009 | .970 ± .237 | -2.396 ± .167 | .290 ± .167 | .485 ± .033 | .715 ± .033 |
| 8 | 2.380 ± .317 | 2.464 ± .237 | 1.536 ± .237 | .012 ± .001 | .880 ± .168 | -3.221 ± .161 | .350 ± .190 | .494 ± .081 | .733 ± .057 |

## Trend evidence and questions

Across severity-level means, the Spearman association with support `N4` is
`+.999` for mean p4 and mean `z4-z3`, and `+.800` for predictive mean; all five
seed-specific slopes for p4 and the margin are positive versus `log(N4)`.
Thus lower support consistently reduces direct class-4 probability and worsens
the C4-vs-C3 classifier margin.

The requested broader localization chain is mixed. Aggregate support
correlations are `-.718` for C4 MAE, `-.800` for shrinkage, and `+.900` for
raw feature-nearest-4 fraction, but individual seed slopes conflict for all
three. C4 MAE is already high at N4=66, improves slightly at 50, then worsens
at 33/16 and remains non-monotonic at 8. Severe C4 error and raw geometry are
also non-monotonic. Class-0 MAE (`.478–.507`) and global L1 MAE
(`.704–.733`) vary modestly without a consistent dose pattern.

1. **Q1–Q2:** only partial evidence: MAE and shrinkage generally worsen below
50 support but are not monotonic or consistently seed-paired.
2. **Q3:** p4 and predictive mean move inward as support falls; routing and
adjacent/severe behavior are noisier.
3. **Q4:** yes, the C4-vs-C3 margin deteriorates strongly and consistently.
4. **Q5:** raw feature-nearest-4 declines in aggregate but is seed-variable;
nearest-centroid assignment remains descriptive and its class-4 centroid is
increasingly noisy at low support.
5. **Q6–Q7:** the probability/margin response is seed-consistent; full
localization and representation response are not. There is no comparable
lower-endpoint or global instability trend.
6. **Q8:** no. The manuscript may report this controlled RetinaMNIST CE
partial result, but must not claim that reduced support systematically
strengthens every form of inward localization.

## Scope and stop condition

No RPS replication, balancing mitigation, Solar run, method design, additional
seed/severity, checkpoint retuning, commit, or push occurred. Complete
machine-readable artifacts are in
`outputs/retinamnist/phase3_20a_imbalance_severity_dose_response/`.
