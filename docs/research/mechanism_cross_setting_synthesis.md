# Mechanism Cross-Setting Synthesis and Claim Lock

**Scope:** frozen synthesis of completed evidence only.  This document does
not retrain a backbone, rerun A/C or H1/H2, alter the frozen protocol, or add a
new experiment.  RetinaMNIST and Solar are the confirmatory evidence block;
UTKFace remains historical supporting evidence only.

## 1. Executive decision

**Claim-lock decision: move to manuscript writing with a bounded mechanism
story.**  The primary empirical result is strong: direction-only frozen-head
adaptation improves rare-end L1 localization in all four predeclared
dataset-objective settings (RetinaMNIST CE/RPS and Solar CE/RPS), for all four
confirmatory backbone seeds in each setting.  The endpoint-geometry result is
not universal: it is strong in Solar, partial in Retina CE, and unsupported by
the preregistered H2a criterion in Retina RPS.

The paper should therefore lead with reproducible rare-end inward localization
and its controlled head-level response, and present endpoint-specific geometry
as a setting-dependent explanatory diagnostic rather than a universal recovery
mechanism.  It must not present balanced sampling, classifier direction, or
generic representation/head decoupling as a new method or general novelty.

## 2. Confirmatory evidence matrix

| Claim component | Retina CE | Retina RPS | Solar CE | Solar RPS | Overall status |
| --- | --- | --- | --- | --- | --- |
| **H1:** C improves rare-end localization over A | Replicated: 4/4 seed signs; mean ΔMAE -0.6553 | Replicated: 4/4; -0.4280 | Replicated: 4/4; mean ΔMAE -0.923 | Replicated: 4/4; -1.031 | **STRONG** across four settings |
| **H2a:** endpoint-specific geometry adds to A-state + generic geometry for final C localization | Partial: beneficial coefficient 4/4, positive ΔLOO 2/4 | Not supported: beneficial coefficient 2/4 | Strong: beneficial coefficient 4/4; ΔLOO positive 4/4 | Strong: beneficial coefficient 4/4; ΔLOO positive 4/4 | **SETTING-DEPENDENT** |
| Cross-objective direction response | Confirmed on CE | Confirmed on RPS | Confirmed on CE | Confirmed on RPS | **STRONG** within the two tested objectives |
| Cross-dataset direction response | RetinaMNIST | RetinaMNIST | Solar | Solar | **STRONG** for two datasets, not universal |
| Geometry mechanism generality | Partial | Unsupported | Strong | Strong | **Not general**; Solar-supported diagnostic |
| Balanced-sampling factor | Historical one-backbone factorial supports balanced C/F over natural E/G | Same frozen RPS representation; objective differences small within strata | Not tested as a confirmatory factorial | Not tested as a confirmatory factorial | **Retina-specific supporting mechanism evidence** |
| Rare-support severity | CE grid: p4 and C4-vs-C3 margin decline with lower support; MAE/shrinkage/severe/routing non-monotonic | Not separately tested | Not tested | Not tested | **PARTIAL / non-monotonic boundary** |
| Mixed representation/head failure | Historical endpoint-like versus representation-inward groups; head-recoverable subset | Historical RPS diagnostics | Historical frozen-head diagnostics | Historical frozen-head diagnostics | **Historical/supporting decomposition**, not universal causal proof |

The four confirmatory seeds are independently trained backbone realizations
within each dataset-objective setting.  They are not 16 independent datasets or
16 independent experimental settings.

## 3. H1 final verdict

**H1 is strongly replicated across the four predeclared settings.**  In every
new seed 1–4, C reduced exact-L1 rare-end MAE relative to A for Retina CE,
Retina RPS, Solar CE, and Solar RPS.  The most defensible frozen statement is:

> **Across independently trained CE- and RPS-based representations in
> RetinaMNIST and Solar, direction-only classifier adaptation reproducibly
> improves rare-end localization under the tested frozen protocols.**

This wording is fully supported.  “Independently trained” modifies backbone
realizations, not datasets; the finding is restricted to the two datasets,
objectives, architectures, splits, and A/C intervention protocol studied.

## 4. H2a final verdict

**H2a is setting-dependent, not a cross-setting mechanism law.**  Solar CE and
Solar RPS meet the strong frozen criterion.  Retina CE is partial because only
two of four seeds have positive incremental leave-one-out value despite the
beneficial coefficient sign in all four; Retina RPS is not supported because
only two of four seeds have the beneficial coefficient sign.

Frozen statement:

> **Endpoint-specific representation geometry can add information about final
> post-adaptation rare-end localization beyond original-head state and generic
> centroid geometry, but this diagnostic is setting-dependent: strong in Solar,
> partial in Retina CE, and unsupported under the preregistered criterion in
> Retina RPS.**

H2b remains secondary and cannot rescue or strengthen this H2a verdict.

## 5. C1–C5 claim audit

| Claim | Final status | Exact evidence | Allowed wording | Forbidden wording |
| --- | --- | --- | --- | --- |
| **C1:** rare upper extremes exhibit inward localization | **STRONG** | Historical RetinaMNIST, Solar, and UTKFace diagnostics; H1 confirms outward response in the two-dataset confirmatory block | “Rare upper-end samples localized inward in the studied imbalanced ordinal settings.” | “All imbalanced ordinal classifiers shrink upper extremes inward.” |
| **C2:** failure is mixed between representation-inward and head-mislocalized samples | **HISTORICAL/SUPPORTING ONLY** | Historical nearest-centroid/head diagnostics in RetinaMNIST and Solar; seed-0 supporting analyses | “Historical diagnostics identify both representation-inward cases and endpoint-like cases that remain head-mislocalized.” | “Mixed failure is a universally replicated causal decomposition.” |
| **C3:** direction-only adaptation is the most consistent actionable head mechanism | **STRONG** | H1 replicated 4/4 backbone seeds in all Retina/Solar CE/RPS settings, with norms/biases/backbones fixed | “Direction-only adaptation is the most consistently reproduced head-level localization response among the tested controls.” | “Direction-only adaptation is a novel or universally optimal classifier correction.” |
| **C4:** balanced sampling is the dominant adaptation factor in Retina factorial | **SUPPORTED WITH QUALIFICATION** | One frozen Retina RPS backbone: balanced C/F improve C4 MAE and margin over natural E/G; within-stratum CE/RPS differences are small | “In the controlled one-backbone Retina factorial, balanced sampling was the dominant observed training signal.” | “Balanced sampling is universally causal or a new method.” |
| **C5:** reduced rare support weakens evidence but lacks a clean monotonic localization dose response | **PARTIAL** | Predeclared 25-run Retina CE grid: p4 and C4-vs-C3 margin deteriorate, whereas C4 MAE, shrinkage, severe burden, and routing are non-monotonic | “Reduced C4 support weakened direct class-4 evidence without yielding a clean monotonic localization dose response in this grid.” | “Rare support alone determines localization severity by a monotonic law.” |

## 6. Robust result versus explanatory diagnostic

This is the cleanest paper hierarchy:

1. **Tier 1 — robust empirical effect:** direction-only adaptation improves
   rare-end localization across the four frozen confirmatory settings.
2. **Tier 2 — explanatory diagnostic:** endpoint-specific centroid geometry
   sometimes predicts final C localization beyond the specified controls.
3. **Tier 3 — boundary:** the geometry diagnostic is not universal, as shown by
   the partial Retina CE and unsupported Retina RPS H2a results.

This hierarchy makes the setting dependence a credibility strength: it avoids
converting a useful diagnostic into an unsupported universal mechanism.

## 7. Novelty boundary

The defensible contribution is a reproducible, ordinal-specific empirical
characterization and controlled decomposition—not a new classifier method.

> **We characterize rare-end inward localization in imbalanced ordinal
> classification and show, across the tested CE/RPS representations in
> RetinaMNIST and Solar, that a fixed-norm, fixed-bias direction-only head
> intervention can recover part of this localization error without changing the
> representation.**

The phrase **“geometry-conditioned recoverability”** is **supporting diagnostic
only**, not central novelty.  It is a strong finding in Solar but is not
established across all four settings.  The paper may state its setting-specific
results, but must not make it the abstract-level general explanation.

## 8. Recommended manuscript framing

### Abstract-level main claim

> We identify rare-end inward localization in the studied imbalanced ordinal
> classifiers and show that a direction-only frozen-head intervention
> reproducibly improves rare-end localization across CE- and RPS-trained
> representations in RetinaMNIST and Solar.

### Introduction contribution bullet

> We separate frozen representation and classifier-head contributions to
> rare-end localization and use a fixed-norm, fixed-bias direction-only
> intervention as a controlled probe of the head-recoverable component.

### Results H1 paragraph

> Under the frozen A/C protocol, rare-end L1 MAE improved in all four new
> backbone seeds for each of the RetinaMNIST CE, RetinaMNIST RPS, Solar CE, and
> Solar RPS settings.  Thus, the direction-only response replicated across the
> four tested dataset-objective settings; this is an empirical result for the
> studied representations, not a universal correction claim.

### Results H2 paragraph

> Endpoint-specific geometry added predictive value for final C localization in
> both Solar settings, was partial in Retina CE, and was not supported in
> Retina RPS under the preregistered H2a criterion.  Geometry is therefore a
> setting-dependent diagnostic of recoverability rather than a universal
> explanation for rare-end recovery.

### Discussion limitation paragraph

> Our evidence covers two confirmatory datasets, two objectives, and four
> independently trained backbone seeds per setting under fixed protocols.
> Endpoint support and geometry effects vary by setting, and neither the
> observed inward localization nor the head response independently identifies
> class imbalance as the sole cause or establishes a universally effective
> intervention.

### Conclusion sentence

> Rare-end inward localization has a reproducible head-actionable component in
> the tested ordinal settings, while representation-geometry diagnostics are
> informative only in a setting-dependent manner.

## 9. UTKFace role

**Recommendation: include historical seed-0 evidence in the supplement only.**
It can motivate the broader phenomenon and document a prior supporting
direction response, but it is not equivalent to Retina/Solar confirmation:
only seed-0 CE/RPS artifacts exist, seeds 1–4 are absent, and a new supporting
replication would require eight backbones.  Label it **SUPPORTING
REPLICATION/HISTORICAL EVIDENCE**, never primary confirmatory evidence.

## 10. Additional-experiment decision

**A — enough evidence; move to manuscript.**  No further experiment is needed
to support the bounded Tier-1 claim.  The failed generality of H2a is a claim
boundary, not an invitation to search for new geometry metrics or relax gates.
Additional data or benchmarks may be considered only as a separately approved
manuscript-strengthening decision, not as a prerequisite for this claim lock.

## 11. ICLR 2027 readiness

| Dimension | Assessment | Basis |
| --- | --- | --- |
| Empirical robustness | **STRONG** | H1 has the same favorable sign in all four confirmatory seeds in all four dataset-objective settings. |
| Mechanistic novelty | **MODERATE** | The ordinal localization/decomposition framing is distinctive, but component ideas are established and H2a is not universal. |
| Statistical rigor | **MODERATE** | Preregistered per-setting gates, four backbone seeds, and H2 controls are strengths; two datasets and limited rare-end support remain constraints. |
| Breadth | **MODERATE** | Two confirmatory datasets × two objectives are coherent but narrow; UTKFace is historical supporting evidence only. |
| Reviewer-rejection risk | **HIGH** | Reviewers may view the controlled head intervention as a known long-tail technique or ask for broader evidence/theory unless the claim boundary is exceptionally clear. |

**Main remaining weakness:** the strongest geometry-conditioned novelty target
does not generalize across all four settings, so the paper cannot center a
universal geometry explanation.  The coherent ICLR story is a careful mechanism
paper with a robust empirical effect and an explicit diagnostic boundary, not a
new state-of-the-art method claim.

## 12. Frozen wording for claims

- “Direction-only classifier adaptation reproducibly improves rare-end
  localization across independently trained CE- and RPS-based representations
  in RetinaMNIST and Solar under the tested frozen protocols.”
- “Endpoint-specific geometry is a setting-dependent diagnostic: strong in
  Solar, partial in Retina CE, and unsupported in Retina RPS under H2a.”
- “Balanced sampling was the dominant observed factor in the controlled,
  one-backbone Retina factorial.”
- “UTKFace is historical/supporting evidence and is excluded from the main
  confirmatory block.”

## 13. Forbidden wording

- “Direction-only adaptation is a new method,” “classifier direction is novel,”
  or “balanced sampling is novel.”
- “Endpoint-specific geometry explains recovery across datasets” or “geometry
  conditioned recoverability is universal.”
- “Sixteen independent experiments” or any wording that treats seeds as
  independent datasets.
- “Class imbalance causes inward localization,” “support has a monotonic dose
  response,” or “RPS is universally superior.”
- “UTKFace confirms the main mechanism” or any elevation of its seed-0 results
  to the primary confirmatory block.
