# Phase 3.17 — Paper Framing and Novelty Boundary

## Scope and evidence ledger

This is an analysis-only paper-framing phase. It uses the completed
RetinaMNIST, UTKFace, and Solar records, especially Phases 3.14–3.16, and the
project's prior literature-overlap audit. No model, feature cache, split, or
new literature search was used. The paper's primary evidence is therefore the
completed frozen-RPS, single-seed mechanism suite: RetinaMNIST training-only
five-fold OOF, archived UTKFace validation, and the already-authorized Solar
archived confirmatory readout. These are mechanistic comparisons, not claims
of architecture-, objective-, or population-universal causality.

The established phenomenon remains distinct from its candidate explanation:

\[
\boxed{\text{Rare upper-extreme inward localization bias under ordinal imbalance}}
\]

The completed evidence supports the following bounded mechanism statement:

> Across the tested frozen RPS representations on RetinaMNIST, UTKFace, and
> Solar, rare upper-end localization consistently improves under classifier
> direction adaptation, while scale, bias, global, and opposite-endpoint
> effects are dataset-dependent.

## Final research question and thesis

**Preferred research question:**

> How do rare upper extremes localize in the studied imbalanced ordinal
> settings, and what frozen classifier-head mechanisms explain the recoverable
> portion of that failure?

This is specific to ordinal error direction and endpoint localization; it is
not a generic tail-accuracy, calibration, or classifier-rebalancing question.

**Central thesis:**

> Across the studied imbalanced ordinal settings, rare upper extremes are often
> localized inward rather than only misclassified; across three datasets, this
> recoverable component is consistently responsive to classifier-direction
> adaptation even though scale, bias, and global predictive effects vary by
> dataset.

## Novelty boundary

### What the paper claims

1. It identifies and characterizes recurring rare-upper-end inward
   localization in imbalanced ordinal settings using routing, ordinal distance,
   predictive mean, shrinkage, endpoint probability, and L1-decision evidence.
2. It documents the phenomenon across RetinaMNIST, UTKFace, and Solar, while
   preserving the different split roles and avoiding raw cross-dataset metric
   rankings.
3. It shows that frozen representations can retain useful rare-end information
   that an original classifier head fails to localize; representation collapse
   remains a separate limitation for part of the failure.
4. It provides controlled frozen-head evidence that classifier direction is a
   consistently supported control axis for rare-end localization across the
   evaluated datasets, while scale, bias,
   lower-endpoint effects, and global behavior are not invariant.
5. It separates local ordinal localization improvement from global predictive,
   calibration, uncertainty, and decision-risk improvement.

### What the paper does not claim

- No universal new classifier correction or final frozen candidate.
- No novelty for balanced-head training, cRT-style retraining, LWS,
  tau-normalization, logit adjustment, weight balancing, or norm control.
- No universal benefit from controlled scale, no transferable `alpha=.50`, and
  no universal causal role for classifier bias.
- No guarantee that head adaptation repairs representation collapse.
- No universal superiority of RPS over CE.
- No universal improvement in global accuracy, calibration, ordinal risk, or
  selective prediction after rare-end localization correction.
- No new conformal-prediction method and no claim that ROP is a robust method
  component.

### What remains open

- A principled direction-targeted intervention that can improve rare-end
  localization without global degradation.
- When classifier scale should be increased, fixed, or reduced.
- How head direction interacts with representation geometry, including which
  representation-collapsed samples are recoverable.
- A theoretical characterization of inward localization under ordinal
  imbalance, and broader multi-seed/domain validation.

## Relationship to adjacent literature

### Generic long-tail recognition

The audited long-tail literature already contains decoupled classifier
retraining (cRT), learned weight scaling (LWS), tau-normalization, logit
adjustment, weight balancing, and balanced classifier retraining. This paper
does not claim that balancing a classifier improves a tail class. Its distinct
empirical object is the **ordinal direction of the error**: a rare upper
endpoint is routed inward toward central classes, with measurable reduction in
predictive mean and positive inward shrinkage. The controlled A-to-C comparison
holds frozen features, original norms, and original biases fixed, so it tests
directional classifier geometry rather than frequency compensation alone.

The paper can therefore contrast rare-end localization, lower/upper endpoint
asymmetry, exact L1 decisions, ordinal distance, predictive mean, shrinkage,
and ordinal decision risk with nominal tail accuracy. It must not present
balanced CE, direction constraints, or scale control as an algorithmic novelty.

### Ordinal imbalance and representation methods

The audited ordinal literature includes imbalance-aware optimization (including
KCOC), ordering/ranking and contrastive representation approaches (including
CORE, CLOC, ConOrd, and PCOR-Net), and related ordinal representation methods.
Those lines broadly aim to improve ordinal prediction, ordering, or
representation structure. This paper instead diagnoses and decomposes a
specific rare-extreme failure mode. Conservatively: in the audited literature,
we did not identify work that explicitly isolates the recoverable frozen-head
component of rare-upper-end inward localization through matched direction,
scale, and bias controls. That is a scoped audit observation, not a claim that
no related work exists.

### Uncertainty and decision risk

Uncertainty is **secondary analysis and motivation**, not the core framing.
Ordinal risk scores can identify severe errors, but risk quality does not by
itself repair localization; conversely, localization correction does not imply
better global uncertainty or risk quality. RPS remains a useful probabilistic
ordinal baseline, but it is not universally superior to CE. This separation is
important evidence and a limitation, not a second method contribution.

## Contribution statements

1. **Phenomenon:** Characterize rare upper-extreme inward localization bias
   under ordinal imbalance using ordinal routing, probability, and predictive-
   location diagnostics across three domains.
2. **Mechanism:** Separate representation limitations from frozen-head
   actionability and show that classifier-direction adaptation is the most
   consistently supported control axis for recoverable rare-end localization
   across the evaluated datasets.
3. **Controlled evidence:** Compare unrestricted balancing, direction-only,
   and controlled-scale heads while reporting endpoint, global, probability,
   and decision-risk outcomes, establishing that localization gains do not
   entail universal global or UQ gains.

## Paper type decision

\[
\boxed{\text{A — MECHANISM / EMPIRICAL ANALYSIS PAPER}}
\]

The central contribution is a reproducible phenomenon and decomposition, not a
validated universal intervention. Calling this a method paper would overstate
the status of established balanced-head tools and the held, non-freeze-ready
combined candidate.

## Title directions

Ranked top three:

1. **Rare-Extreme Localization Bias in Imbalanced Ordinal Classification**
2. **Classifier Head Geometry and Rare-Extreme Localization under Ordinal Imbalance**
3. **Understanding Rare-End Localization Failure in Ordinal Classification**

Additional candidates:

4. **When Rare Ordinal Extremes Collapse Inward: A Cross-Dataset Head Analysis**
5. **Beyond Tail Accuracy: Inward Localization of Rare Extremes in Ordinal Classification**

## Abstract skeleton

1. **Problem/gap:** Ordinal imbalance is usually assessed through accuracy or
   aggregate ordinal error, which can miss directional rare-end failures.
2. **Phenomenon:** Define and observe rare upper-extreme inward localization
   bias: predictions and predictive means for rare upper endpoints shift toward
   interior classes.
3. **Mechanism analysis:** Separate frozen representation and head effects,
   then control classifier direction, scale, and bias to identify the
   recoverable head component.
4. **Evidence:** Report matched frozen-RPS evidence from RetinaMNIST, UTKFace,
   and Solar; direction adaptation consistently improves rare-end localization,
   whereas scale, bias, global, and opposite-endpoint effects vary.
5. **Implication:** Rare-end localization requires distinct diagnosis from
   global predictive or uncertainty quality; generic head correction is not a
   universally validated solution.

## Introduction plan

1. **Ordinal classification and imbalance.** Explain why a one-class inward
   error differs materially from nominal tail-class failure and why extremes
   have asymmetric decision meaning.
2. **Missing failure mode.** Motivate why accuracy, MAE, QWK, and class recall
   alone obscure directional collapse of a rare endpoint toward the center.
3. **Phenomenon.** Define rare upper-extreme inward localization with routing,
   predictive mean, and inward shrinkage; state the three-domain observation.
4. **Mechanism question.** Separate representation from head geometry and
   introduce direction, scale, and bias as controlled classifier components.
5. **Main findings.** Summarize frozen-head actionability, cross-dataset
   direction evidence across the evaluated settings, and dataset-dependent
   scale/bias/global consequences.
6. **Contributions.** State the three contributions above and the boundary:
   analysis rather than a universal corrective method.

## Proposed paper structure

1. Introduction
2. Related Work
   - Imbalanced and long-tail classification
   - Ordinal classification and ordinal imbalance
   - Ordinal uncertainty, calibration, and decision risk
3. Problem Setup and Localization Diagnostics
4. Rare-Extreme Inward Localization under Ordinal Imbalance
5. Frozen Classifier-Head Mechanism Analysis
6. Cross-Dataset Mechanism Evidence
7. Discussion and Limitations
8. Conclusion

A conventional “Method” section would be misleading. Sections 3 and 5 should
be titled **diagnostic protocol** and **mechanism analysis**, respectively.

## Core figures and tables

| Item | Must show | Evidence role |
| --- | --- | --- |
| Figure 1 — Conceptual routing | A rare upper endpoint routed toward central ordinal classes; predictive mean and shrinkage notation | Defines the failure mode without claiming a learned method. |
| Figure 2 — Three-dataset localization | A/B/C/D routing or predictive-mean/shrinkage summaries with within-dataset arrows | Establishes the phenomenon and direction response without comparing raw scales across datasets. |
| Figure 3 — Head decomposition | Original, balanced, direction-only, and controlled-scale heads; direction/scale/bias held or changed | Makes the controlled causal-style comparisons and non-universal scale result legible. |
| Table 1 — Dataset and endpoint support | Task, split role, ordinal classes, rare-end counts, imbalance profile | Makes denominator and provenance differences explicit. |
| Table 2 — Rare-end localization | Exact routing, rare MAE, pK, predictive mean, shrinkage, severe error for A/B/C/D | Separates endpoint localization from global metrics. |
| Table 3 — Mechanism disposition | Direction, scale, bias, head actionability, representation contribution, global/risk outcomes | Prevents overclaiming and records dataset-dependent effects. |

## Terminology audit

| Concept | Canonical wording | Avoid |
| --- | --- | --- |
| Central phenomenon | **rare upper-extreme inward localization bias** | “center shrinkage” as the primary label; it is less specific. |
| Endpoint | rare upper endpoint | alternating casually between tail, extreme, endpoint, and class 4/X. |
| Location movement | inward localization / outward localization | “misclassification direction” without ordinal context. |
| Continuous diagnostic | inward shrinkage | “bias” when referring only to predictive mean. |
| Endpoint comparison | endpoint asymmetry | “fairness asymmetry” unless fairness is studied. |
| Unit classifier vector | classifier direction | “weight” when scale is intentionally factored out. |
| Row magnitude | classifier scale | “norm correction” as a method name. |
| Frozen-head effect | head actionability | “head failure” for every sample. |
| Aggregate result | global predictive quality | “overall performance” when the metric is specific. |
| Risk measure | ordinal decision risk | generic “uncertainty” when referring to Bayes L1 risk. |

## Claim-strength audit

| Claim | Evidence strength | Allowed wording | Avoid |
| --- | --- | --- | --- |
| Inward rare-upper localization occurs in the studied settings | Strong across three datasets | “observed across the three evaluated datasets” | “universal consequence of imbalance” |
| Frozen heads are actionable | Strong across three datasets | “can materially move head-recoverable rare-end localization” | “head adaptation solves the failure” |
| Direction adaptation helps rare-end localization | Strong within the tested frozen-RPS protocols | “consistently improved rare-end localization in our controls” | “is the exclusive cause” |
| Scale helps | Mixed | “modulates correction strength in a dataset-dependent manner” | “scale is required” or “alpha=.50 transfers” |
| Bias matters | Mixed/causally unsupported outside Retina swaps | “bias movement is dataset-dependent” | “bias correction is effective” |
| Representation limits exist | Partial | “representation collapse explained a subset in audited datasets” | “all failures originate in the representation” |
| Global prediction improves | Unsupported | “localization and global quality can diverge” | “the intervention improves performance” |
| UQ/risk improves | Unsupported as a general claim | “risk metrics can disagree with localization” | “localization fixes uncertainty” |
| Opposite endpoint is harmed | Dataset-specific | “endpoint collateral effects varied by dataset” | “rare-end correction harms class 0” |
| Beyond studied datasets | Limited | “within three frozen-RPS evaluations” | “general ordinal law” |

## Limitations

- The mechanism suite covers three datasets and one backbone seed per dataset;
  it is not a multi-seed, architecture-universal study.
- All direct mechanism tests use frozen RPS representations; conclusions about
  other objectives or end-to-end adaptation are out of scope.
- Controlled head interventions provide stronger evidence than correlation but
  do not establish a theoretical causal account of ordinal imbalance.
- No universal corrective method or performance guarantee is proposed.
- Solar, UTKFace, and RetinaMNIST differ substantially in domain, split role,
  and endpoint support; comparisons are qualitative within-dataset effects.
- Direction adaptation can improve rare-end localization while global metrics,
  probability quality, or ordinal risk worsen.

## Venue fit

| Venue | Fit | Rationale |
| --- | --- | --- |
| Pattern Recognition | **1** | Accommodates rigorous diagnostic empirical work spanning medical imaging and a scientific forecasting domain, without requiring a new universal algorithm. |
| IEEE BigData or a comparable applied data-mining venue | **2** | Fits the cross-domain empirical diagnosis, imbalance, and decision-risk framing if the final manuscript foregrounds reproducible evidence and practical analysis. |
| ICLR | Lower | A mechanism result could be interesting, but lack of a new general learning method or theory makes fit difficult. |
| NeurIPS | Lower | The topic aligns with reliability and long-tail concerns, but a broader methodological or theoretical contribution would usually be expected. |
| ICML | Lower | Similar concern: the present strongest contribution is empirical diagnosis rather than a new general learning algorithm. |

## Final framing decision

\[
\boxed{\text{A — PROCEED AS MECHANISM PAPER}}
\]

The phenomenon, frozen-head actionability, and three-dataset direction evidence
form a coherent bounded paper narrative. The manuscript should foreground the
diagnosis and mechanism boundary, not present a balanced head, scale rule, or
ROP term as a novel solution. The next authorized work is manuscript planning
and writing from the completed evidence; any new scientific experiment still
requires separate authorization.
