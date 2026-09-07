# Phase 3.14 — Cross-Dataset Mechanism Consolidation

## Scope and evidence ledger

This is an analysis-only consolidation of the completed frozen-feature
RetinaMNIST and UTKFace evidence. No model was trained, no feature was
extracted, no split was evaluated, and no test artifact was opened. RetinaMNIST
numbers are pooled five-fold training-only OOF evidence; UTKFace numbers are a
single archived train/validation seed-0 confirmation. They are therefore
compared for direction of effect, not raw magnitude or formal significance.

The established phenomenon remains separate from the head-mechanism claim:

\[
\boxed{\text{Rare upper-extreme inward localization bias under ordinal imbalance}}
\]

It is supported by baseline evidence on RetinaMNIST, UTKFace, and Solar. The
mechanism synthesis below concerns only the frozen RPS-head A/B/C/D controls on
RetinaMNIST and UTKFace.

## Cross-dataset evidence matrix

All values below use exact L1 Bayes decisions. A is original RPS, B is the
unrestricted balanced head, C is fixed-original-norm direction-only training,
and D is direction-only training with the predeclared intermediate scale
`alpha=.50`. A dash means that the cited phase note did not report that field
for D; it is not inferred.

### Rare upper endpoint

| Dataset / condition | C4 MAE | Exact C4 | Mean p4 | Predictive mean | Shrinkage | C4 severe |
|---|---:|---:|---:|---:|---:|---:|
| Retina A | 1.697 | 0/66 | .116 | 2.203 | 1.797 | 65.2% |
| Retina B | 1.121 | 11/66 | .338 | 2.727 | 1.273 | 25.8% |
| Retina C | 1.348 | 2/66 | .306 | 2.616 | 1.384 | 33.3% |
| Retina D | 1.258 | 8/66 | — | — | — | — |
| UTKFace A | .6119 | 35/67 | .4988 | 3.2454 | .7546 | 10.4% |
| UTKFace B | .5224 | 38/67 | .5389 | 3.4080 | .5920 | 6.0% |
| UTKFace C | .5075 | 39/67 | .5766 | 3.4181 | .5819 | 6.0% |
| UTKFace D | .4925 | 40/67 | .5914 | 3.4497 | .5503 | 6.0% |

On both datasets, B and C improve rare-end localization over A. Fixed original
scale C preserves a real portion of the gain but loses much of Retina's exact
recovery (`2` versus B's `11`). Additional intermediate scale D restores much
of that loss on Retina (`8`) and is the strongest UTKFace localization point
(`.4925` C4 MAE). The D-versus-C increment is larger on Retina than UTKFace,
but its direction is consistent.

### Lower endpoint and global behavior

| Dataset / condition | C0 MAE | C0 severe | Global L1 MAE | QWK | Probability / risk qualification |
|---|---:|---:|---:|---:|---|
| Retina A | .570 | 25.1% | .696 | .604 | NLL/Brier/RPS `.1.141/.554/.122`; Spearman `.467` |
| Retina B | .733 | 24.3% | .744 | .612 | probability quality worsens; Spearman `.471` |
| Retina C | .691 | 22.0% | .721 | .621 | improves B global/C0 cost; Spearman `.447` |
| Retina D | .698 | 22.4% | .731 | .619 | improves B NLL/Brier/RPS; mixed risk metrics |
| UTKFace A | .3268 | .65% | .2691 | .8191 | NLL/Brier/RPS `.658/.356/.0498`; Spearman `.382` |
| UTKFace B | .1808 | 1.53% | .2737 | .8362 | improves NLL/Brier and all reported risk metrics |
| UTKFace C | .1895 | 2.83% | .3138 | .8209 | global/probability quality worse than B |
| UTKFace D | .1656 | 2.18% | .3096 | .8235 | best C0 MAE, but global/probability quality worse than B |

Retina's balanced correction redistributes error harmfully toward class 0 and
worsens global MAE. UTKFace instead improves C0 MAE under every adapted head;
however, C/D still incur a substantial global-MAE and probability-quality cost
relative to B. Thus neither endpoint safety nor global quality has a universal
ordering under correction strength.

### Parameter mechanism

| Component | RetinaMNIST | UTKFace | Cross-dataset reading |
|---|---|---|---|
| Direction rotation | Large B rotations; C4 B-vs-A cosine `.314` (about 72 degrees) | Large B rotations; C4 B-vs-A cosine `.250` | Direction changes are material in both. |
| Norm/scale increase | C4 norm `.575 -> 3.285`; removing it loses most exact recovery | C4 norm `.688 -> 6.634`; D's intermediate norm is `3.661` | Large scale amplification accompanies stronger outward movement in both. |
| Direction-only C | Improves C4 MAE A→C `1.697->1.348` | Improves C4 MAE A→C `.6119->.5075` | Direction adaptation is independently useful. |
| Intermediate D | Restores C4 recovery C→D `2->8`, safer than B in OOF | Improves C4 MAE C→D `.5075->.4925`, but globally worse than B | Controlled scale diagnoses correction strength; it is not a universal optimum. |
| Bias shift | `.003-.008`, swaps inert | C0/C4 shifts `-.061/+ .108` | Bias behavior is dataset-dependent; UTKFace shifts have no causal swap evidence. |

Conceptually, the supported head decomposition is

\[
w_k=s_kv_k+b_k,
\]

where both direction \(v_k\) and scale \(s_k\) show cross-dataset evidence
for rare-end movement. The contribution of bias \(b_k\), and the decision-risk
consequences of all three terms, remain dataset-dependent.

## Claim disposition

| Claim / component | Cross-dataset status | Final wording |
|---|---|---|
| Rare upper-end inward bias | **TRANSPORTED** | The rare upper endpoint is inward-localized under imbalance across the established RetinaMNIST, UTKFace, and Solar baselines. |
| Head actionability | **TRANSPORTED** | Frozen balanced-head adaptation can move a head-recoverable rare endpoint outward on RetinaMNIST and UTKFace. |
| Direction adaptation | **TRANSPORTED** | With original scale/bias held fixed, direction-only adaptation improves C4 localization over A on both datasets. |
| Fixed original scale is too restrictive | **PARTIALLY TRANSPORTED** | It is clearly too restrictive for Retina exact recovery; UTKFace C is already strong but D still improves C4 MAE. |
| Scale amplification | **PARTIALLY TRANSPORTED** | Larger/intermediate scales accompany and can strengthen rare-end movement on both datasets, but necessity is causally strongest on Retina. |
| Controlled scale | **PARTIALLY TRANSPORTED** | It is evidence about useful correction strength, not a universally superior classifier or a novel standalone method. |
| Opposite-endpoint damage | **DATASET-SPECIFIC** | B damages Retina C0 MAE but improves UTKFace C0 MAE; this is not a universal cost. |
| Global trade-off | **PARTIALLY TRANSPORTED** | Adaptation changes global quality on both datasets, but the ordering differs: Retina D improves B's OOF global MAE, whereas UTKFace D is worse than B/A. |
| Bias contribution | **DATASET-SPECIFIC** | Retina biases are causally inert; UTKFace has nonzero shifts without causal evidence that a bias-only intervention helps. |
| ROP | **NOT SUPPORTED AS A CROSS-DATASET METHOD COMPONENT** | Preserve it only as a diagnostic/secondary idea: OOF gains reversed or disappeared on Retina validation and no transport evidence exists. |
| Combined alpha=.50/lambda=1 candidate | **HOLD — NOT FREEZE-READY** | Phase 3.11 remained mixed; Phase 3.13 excludes ROP and does not repair its validation failure. |

The project-wide **bias-correction STOP** remains correct. A nonzero UTKFace
bias shift is observational, not a demonstrated causal route to recovery; it
does not justify reviving a bias-only or logit-adjustment branch.

## Correct interpretation of controlled scale and safety

Controlled scale should be described as a causal probe of **correction
strength**. On Retina OOF, a predeclared non-isolated alpha `.50-.75` region
retained much of B's rare-end recovery while moderating B's class-0/global
cost. On UTKFace, D had the best class-4 and class-0 MAE, but did not beat B on
global MAE or probability quality. It therefore supports neither a universal
safety claim nor a generic scale-control algorithm claim.

The apparent safety contradiction has a coherent interpretation: rare-end
correction transports, but its redistribution of errors across the remaining
ordinal support depends on the dataset's feature geometry, class distribution,
and original head. The same is true of global decision and risk behavior. This
is why the evidence must retain the sequence

\[
\text{Representation} \rightarrow \text{head geometry} \rightarrow
\text{ordinal localization} \rightarrow \text{decision/risk behavior}
\]

rather than treating a localization gain as a universal global-UQ improvement.

## Solar decision

Solar already strongly confirms the *phenomenon* and Phase 3.9 establishes
frozen-head actionability among geometrically X-like samples. What is missing
is the causal direction-versus-scale decomposition. A minimal third-domain
confirmation would materially distinguish a cross-domain head-geometry
mechanism from a two-image-benchmark result, especially because Solar has a
much stronger rare-extreme failure and a different domain.

\[
\boxed{\text{A — SOLAR DIRECTION/SCALE CONFIRMATION JUSTIFIED}}
\]

This is scientific justification for a separately authorized future phase, not
authorization for execution here. Its minimal frozen design should be A original
Solar RPS head, B balanced head, C direction-only fixed-original-scale head, and D fixed
`alpha=.50` controlled-scale head. It must use the existing Solar split role
only if that role is predeclared appropriately; include no ROP, alpha grid,
new loss, representation training, or test-driven adaptation.

## Project-level mechanism statement

> **Across RetinaMNIST and UTKFace, rare upper-end localization responds
> consistently to classifier direction adaptation, with additional scale
> enabling stronger outward movement; the resulting global, opposite-endpoint,
> and bias effects are dataset-dependent.**

This is a mechanistic observation, not a novel generic scale-control method,
not a final candidate validation, and not a claim of universal performance.
