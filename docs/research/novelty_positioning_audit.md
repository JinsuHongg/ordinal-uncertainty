# Novelty Positioning Audit

**Status:** Working literature audit for ICLR 2027 positioning  
**Project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Last updated:** 2026-09-13

## 1. Purpose

This document tests whether the current mechanism story is scientifically distinct enough to justify additional replication experiments and an ICLR submission.

The objective is not to maximize the number of claimed contributions. The objective is to identify one central claim that:

1. is supported by the current evidence;
2. is not already established by closely related long-tailed or ordinal-classification work;
3. can be strengthened with a bounded replication study;
4. does not require presenting classifier rebalancing, balanced sampling, weight normalization, or direction adaptation as a new method.

Current decision:

> **PROVISIONAL GO — only for a narrow ordinal endpoint mechanism claim.**

The literature already covers representation/classifier decoupling, balanced classifier retraining, classifier weight-norm imbalance, angular decision geometry, long-tail representation geometry, and ordinal imbalance. The remaining opportunity is the combination of **ordinal endpoint localization, representation/head mismatch, and geometry-conditioned recoverability**.

---

## 2. Current empirical phenomenon

Across RetinaMNIST, UTKFace, and Solar, rare upper extremes show a recurring inward-localization pattern in the studied imbalanced ordinal settings.

The strongest observed cases are RetinaMNIST and Solar:

- rare upper-end decisions concentrate on adjacent or interior classes;
- the predictive mean is shifted inward from the true endpoint;
- upper-end class-conditional L1 error is substantially larger than the corresponding lower-end error;
- exact rare-end recovery can be absent under the original head even when some rare-end samples retain endpoint-aligned feature geometry.

The bounded descriptive statement is:

> **In the studied imbalanced ordinal settings, rare upper extremes exhibit systematic inward localization.**

This statement is descriptive and is not, by itself, sufficient novelty.

It does **not** imply:

- class imbalance alone causes the phenomenon;
- all ordinal datasets exhibit it;
- upper-end error direction itself is novel;
- the representation has collapsed whenever a sample is not nearest to the rare-end centroid.

---

## 3. Triviality boundary: what is not novel about "inward" errors

For the maximum ordinal class, every incorrect discrete class prediction is necessarily inward. Therefore, the observation

> rare upper classes are misclassified toward lower classes

cannot serve as the central contribution.

The nontrivial quantities are instead:

- **error magnitude** along the ordinal axis;
- **predictive-location shrinkage** from the endpoint;
- **rare-vs-adjacent probability/logit competition**;
- **lower-vs-upper endpoint asymmetry**;
- whether endpoint-aligned **feature geometry is retained**;
- whether the classifier head fails to use retained endpoint geometry;
- which samples remain **recoverable** under a controlled head intervention;
- whether recoverability varies systematically with a **continuous representation margin**.

The paper should be framed around these quantities, not around the sign of the upper-end error.

---

## 4. Closest prior-work families

### 4.1 Representation/classifier decoupling in long-tailed recognition

**Kang et al., ICLR 2020 — _Decoupling Representation and Classifier for Long-Tailed Recognition_.**

Established:

- representation learning and classifier learning can be separated;
- naturally sampled representations can remain useful under long-tailed training;
- adjusting/retraining only the classifier with balancing strategies can substantially improve tail recognition.

Direct overlap with this project:

- frozen-representation analysis;
- classifier-head retraining;
- balanced sampling at the classifier stage;
- distinction between representation quality and classifier behavior.

Therefore, the following are **not novel claims**:

- classifier retraining helps rare classes;
- balanced classifier retraining can recover tail accuracy;
- representation and classifier contributions can be separated.

What remains potentially distinct:

- ordered endpoint versus adjacent-interior localization;
- probability-location shrinkage rather than only class accuracy;
- sample-level endpoint geometry versus head routing;
- recoverability conditioned on retained ordinal endpoint geometry.

Reference: https://ai.meta.com/research/publications/decoupling-representation-and-classifier-for-long-tailed-recognition/

### 4.2 Classifier weight norms and long-tail bias

**Alshammari et al., CVPR 2022 — _Long-Tailed Recognition via Weight Balancing_.**

Established:

- naive long-tail training can produce strongly imbalanced classifier weight norms;
- weight balancing and regularization can improve tail recognition;
- classifier-head parameters themselves are an important source of long-tail bias.

Direct overlap:

- analysis of per-class classifier weights;
- scale/norm decomposition;
- controlled classifier-stage intervention.

Therefore, the following are **not novel claims**:

- classifier norms differ with imbalance;
- scale/norm control can help tail classes;
- classifier geometry matters under long-tail training.

Potential distinction:

- fixed-norm, fixed-bias **direction-only** intervention;
- rare-end-versus-adjacent ordinal competition;
- localization and recoverability rather than global tail accuracy.

Reference: https://openaccess.thecvf.com/content/CVPR2022/html/Alshammari_Long-Tailed_Recognition_via_Weight_Balancing_CVPR_2022_paper.html

### 4.3 Angular classifier geometry is already prior art

**Wang et al., CVPR 2022 — _C2AM Loss: Chasing a Better Decision Boundary for Long-Tail Object Detection_.**

Established:

- angular classifier geometry and decision boundaries matter in long-tail recognition;
- cosine similarity and category-aware angular margins can modify tail decision regions.

This makes the generic statement

> classifier direction/angle matters for rare classes

insufficient as novelty.

Potential distinction:

- direction is used here as a **mechanistic probe**, not a new margin loss;
- class-specific norms and biases are held fixed;
- the target is ordinal endpoint localization and sample-level recovery.

Reference: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_C2AM_Loss_Chasing_a_Better_Decision_Boundary_for_Long-Tail_Object_CVPR_2022_paper.html

### 4.4 Long-tail representation geometry is already prior art

**Zhu et al., CVPR 2022 — _Balanced Contrastive Learning for Long-Tailed Visual Recognition_.**

Established:

- long-tailed data can distort class geometry in representation space;
- balanced representation learning can restore a more symmetric class configuration.

**Yi et al., ICLR 2025 — _Geometry of Long-Tailed Representation Learning: Rebalancing Features for Skewed Distributions_.**

Established:

- long-tail imbalance can skew learned feature distributions;
- tail-class centers can shrink together or collapse under identifiable conditions;
- representation geometry itself can be a central long-tail failure mode.

Therefore, the following are **not novel claims**:

- tail representations are distorted by imbalance;
- tail-class centroids can become poorly separated;
- feature geometry matters in long-tailed recognition.

Potential distinction:

- some rare-end samples are representation-inward while others remain endpoint-aligned;
- these groups respond differently to the **same controlled head intervention**;
- the relevant geometry is ordinal endpoint versus adjacent interior, not generic tail-class separation.

References:

- https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/adb2075b6dd31cb18dfa727240d2887e-Abstract-Conference.html

### 4.5 Feature prototypes and classifier weights are complementary representations

**Parisot et al., CVPR 2022 — _Long-Tail Recognition via Compositional Knowledge Transfer_.**

Established:

- feature prototypes and learned cosine-classifier weights provide complementary class representations;
- classifier/prototype information can be exploited to improve rare classes.

This is relevant to our use of both class centroids and classifier weights, but it does not establish ordinal endpoint localization or geometry-conditioned recovery.

Reference: https://openaccess.thecvf.com/content/CVPR2022/html/Parisot_Long-Tail_Recognition_via_Compositional_Knowledge_Transfer_CVPR_2022_paper.html

### 4.6 Feature/classifier space alignment in long-tail learning

**Wang et al., AAAI 2026 — _Space Alignment Matters: The Missing Piece for Inducing Neural Collapse in Long-Tailed Learning_.**

Established:

- long-tailed learning can produce pronounced misalignment between feature and classifier-weight spaces;
- such misalignment can harm generalization;
- explicit feature/classifier alignment can improve long-tail recognition.

This is a particularly important close prior because a vague statement such as

> the classifier fails to align with the representation

would be too close to this work.

Required distinction:

- ordered endpoint versus adjacent-interior structure;
- predictive-location shrinkage;
- controlled direction-only intervention with fixed norm/bias;
- sample-level recoverability conditioned on endpoint geometry.

Reference: https://ojs.aaai.org/index.php/AAAI/article/view/39835

### 4.7 Ordinal imbalance is an established problem

**Zhu et al. — _Minority Oversampling for Imbalanced Ordinal Regression_ (SMOR).**

Established:

- imbalanced ordinal data have structure that ordinary oversampling can violate;
- the ordering of classes should be respected when generating minority samples;
- distant ordinal mistakes are more consequential than nearby ones.

**Lázaro and Figueiras-Vidal, Pattern Recognition 2023 — _Neural Network for Ordinal Classification of Imbalanced Data by Minimizing a Bayesian Cost_.**

Established:

- neural ordinal classification can explicitly combine imbalance and ordinal decision costs;
- decision thresholds and network parameters can be optimized for imbalanced ordinal problems.

Therefore, the following are **not novel claims**:

- imbalance is important in ordinal classification;
- order-aware treatment of minority classes is needed;
- distance-aware ordinal errors are important.

Potential distinction:

- deep representation/head decomposition;
- endpoint-specific localization and asymmetry;
- controlled recovery mechanism.

References:

- https://www.sciencedirect.com/science/article/pii/S0950705118306166
- https://www.sciencedirect.com/science/article/pii/S0031320323000043

---

## 5. Paper-by-paper comparison matrix

Legend:

- **Y**: explicit focus/result in the cited work;
- **P**: partial or adjacent overlap;
- **N**: not a central component of the cited work as identified in this audit.

| Work | Imbalance | Ordinal structure | Rep. geometry | Head decomposition | Norm/scale | Direction/angle | Endpoint localization | Endpoint asymmetry | Sample recoverability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kang et al., ICLR 2020 | Y | N | P | Y | P | N | N | N | N |
| Alshammari et al., CVPR 2022 | Y | N | P | Y | Y | P | N | N | N |
| C2AM, CVPR 2022 | Y | N | P | P | Y | Y | N | N | N |
| Balanced Contrastive Learning, CVPR 2022 | Y | N | Y | P | N | P | N | N | N |
| Compositional Knowledge Transfer, CVPR 2022 | Y | N | Y | P | P | P | N | N | N |
| Geometry of LTR, ICLR 2025 | Y | N | Y | P | N | P | N | N | N |
| Space Alignment Matters, AAAI 2026 | Y | N | Y | Y | P | P | N | N | N |
| SMOR | Y | Y | P | N | N | N | N | N | N |
| Lázaro & Figueiras-Vidal, PR 2023 | Y | Y | N/P | N | N | N | P | N | N |
| Current project | Y | Y | Y | Y | Y | Y | Y | Y | Y* |

`Y*` means that sample recoverability is currently supported by binary centroid-group evidence, but the stronger continuous geometry-conditioned claim still requires multi-backbone replication.

---

## 6. Current evidence map

### C1 — Rare upper extremes exhibit inward localization

Evidence:

- RetinaMNIST;
- UTKFace;
- Solar;
- routing;
- predictive mean;
- inward shrinkage;
- lower-versus-upper endpoint L1 error.

Novelty risk:

- **High** if phrased only as "rare upper classes are predicted inward."

Use:

- motivating phenomenon;
- cross-domain empirical observation;
- not sufficient central novelty by itself.

### C2 — Failure is mixed between representation and classifier head

Current evidence:

RetinaMNIST CE:

- 48/66 true class-4 samples are nearest to the rare-end centroid;
- 18/66 are representation-inward;
- the original head recovers none exactly;
- all 11 exact direction-only recoveries occur among rare-end-like samples.

Solar CE:

- 724/921 true X-class samples are rare-end-like;
- 197/921 are representation-inward;
- the original head recovers none exactly;
- all 559 exact direction-only recoveries occur among X-like samples.

Novelty assessment:

- **Promising**, if explicitly ordinal and sample-level.

Important distinction from generic decoupling:

> An ordinal endpoint sample can retain endpoint-aligned feature geometry while the classifier still maps it toward the adjacent interior class.

Current limitation:

- nearest-centroid grouping is a descriptive diagnostic, not a causal definition of representation failure;
- most mechanism evidence is from one backbone realization per setting.

Required strengthening:

- independent backbone-seed replication;
- continuous representation margin.

### C3 — Direction-only adaptation produces a consistent head-level localization response

Current evidence:

- RetinaMNIST RPS: rare-end L1 MAE 1.697 -> 1.348;
- RetinaMNIST CE: 1.439 -> 1.030;
- Solar RPS: 1.197 -> 0.675;
- Solar CE: 1.115 -> 0.586;
- UTKFace RPS also improves.

Novelty assessment:

- **Not novel as "direction matters."**
- Potentially useful as a controlled mechanistic intervention when combined with fixed norm/bias and ordinal endpoint recovery.

Safe interpretation:

> Direction-only adaptation provides a controlled probe showing that endpoint localization can improve without modifying the frozen representation, class-specific scales, or biases.

Required strengthening:

- independent backbone-seed replication.

### C4 — Balanced sampling is the dominant adaptation factor in the controlled RetinaMNIST factorial

Current evidence:

Frozen RPS representation, direction-only head:

- Natural + CE: C4 MAE 1.697, margin -0.912;
- Natural + RPS: C4 MAE 1.712, margin -0.878;
- Balanced + CE: C4 MAE 1.348, margin +0.030;
- Balanced + RPS: C4 MAE 1.318, margin +0.083.

Novelty assessment:

- **Not a main novelty claim.**

Use:

- mechanism evidence for the training signal associated with the beneficial direction change;
- balanced sampling itself is prior art.

If retained as a major result:

- repeat across independent backbone seeds.

### C5 — Rare-class support does not produce a simple localization dose response

Current evidence:

As class-4 support decreases from 66 to 8:

- p4 deteriorates consistently;
- z4-z3 deteriorates consistently;
- L1 MAE is non-monotonic;
- inward shrinkage is non-monotonic;
- severe error is non-monotonic;
- feature-nearest endpoint fraction is seed-variable.

Novelty assessment:

- useful negative/boundary result;
- not sufficient as central novelty.

Scientific role:

> Rare support weakens direct endpoint evidence, but support count alone does not determine final localization severity through a simple monotonic law.

---

## 7. Candidate central claims

### Candidate A — phenomenon only

> Rare upper extremes in imbalanced ordinal classifiers exhibit systematic inward localization.

**Decision:** Reject as central novelty.

Reason:

- an incorrect prediction for the maximum class is necessarily inward;
- prior work already studies extreme-prediction shrinkage and imbalanced ordinal learning in adjacent forms.

Keep as the motivating empirical phenomenon.

### Candidate B — mixed representation/head failure

> Rare-end localization failure can occur at both the representation and classifier-head levels.

**Decision:** Promising but too broad by itself.

Risk:

- generic representation/classifier decoupling is already established.

Required ordinal distinction:

> Some rare-end samples retain endpoint-aligned geometry yet are mapped toward the adjacent ordinal class by the classifier head.

### Candidate C — controlled head recoverability

> Rare-end samples can retain endpoint-aligned feature geometry while the classifier head localizes them inward, and direction-only adaptation can recover a substantial subset without modifying the representation.

**Decision:** **Current GO candidate.**

Why it is stronger:

- explicitly ordinal;
- not reducible to tail accuracy;
- links feature geometry to classifier action;
- direction is a controlled probe rather than a proposed method;
- supported descriptively in RetinaMNIST and Solar.

Main weakness:

- replication across independent backbone realizations is not yet sufficient.

### Candidate D — geometry-conditioned recoverability

> Endpoint-specific representation geometry is informative about direction-only recovery beyond generic sample difficulty and original-head confidence.

**Decision:** **Strongest novelty hypothesis; not yet established.**

Current evidence:

- the existing all-centroid nearest-endpoint subgroup analysis suggests that samples retaining endpoint-like geometry are disproportionately recoverable under direction-only adaptation;
- however, this may partly reflect generic sample difficulty or initial classifier confidence rather than ordinal endpoint structure itself.

Required evidence:

- preserve the existing all-centroid binary subgroup definition exactly;
- add a separately named continuous endpoint-vs-adjacent representation margin;
- distinguish final recovery/localization under C from the A->C change magnitude;
- condition or adjust for original-head localization/confidence and generic class-separation difficulty;
- use independent backbone seeds;
- require consistent evidence across RetinaMNIST and Solar, with CE and RPS treated as separate settings.

This claim should not enter the abstract as an established result until the replication protocol is completed.

---

## 8. Recommended central positioning

### Final positioning after frozen replication

> **We characterize rare-end inward localization in imbalanced ordinal
> classification and show that a fixed-norm, fixed-bias direction-only head
> intervention reproducibly improves rare-end localization across the tested
> CE- and RPS-based representations in RetinaMNIST and Solar, without changing
> the representation.**

This is a controlled empirical mechanism claim, not a new classifier method,
not a claim that direction is novel, and not a universal correction claim.

### Geometry-conditioned recoverability: final status

> **Endpoint-specific representation geometry can add explanatory value for
> final direction-only localization beyond original-head state and generic
> centroid geometry, but this is setting-dependent: strong in Solar, partial in
> Retina CE, and unsupported under the preregistered H2a criterion in Retina
> RPS.**

Geometry-conditioned recoverability is consequently a supporting diagnostic,
not the central novelty or an abstract-level universal mechanism claim.  The
paper may report the setting-specific H2a pattern, but failure to recover must
not be labeled as a representation-limited causal failure.

---

## 9. Claims explicitly ruled out

Do not claim:

- a new classifier-balancing method;
- a new resampling method;
- novelty for classifier direction or angular geometry in general;
- novelty for representation/classifier decoupling in general;
- novelty for long-tail representation distortion in general;
- universal superiority of balanced-head retraining;
- universal causal effect of class imbalance;
- universal RPS superiority;
- universal scale, bias, or norm mechanism;
- representation collapse from nearest-centroid routing;
- irrecoverability of representation-inward samples;
- objective independence;
- monotonic localization degradation with rare-class support.

---

## 10. Remaining literature questions

The first audit is sufficient to justify a bounded replication study, but the Related Work section should continue to watch for papers that directly combine multiple elements below:

1. ordinal long-tail classification with endpoint-specific probability localization;
2. sample-level recovery conditioned on feature geometry;
3. adjacent-class logit margins under ordinal imbalance;
4. classifier-head direction analysis with fixed class-specific norms and biases;
5. explicit lower-versus-upper endpoint asymmetry;
6. probability-location shrinkage or regression-to-the-interior in discrete ordinal classifiers.

For any newly identified close paper, record:

- task and venue;
- ordinal versus nominal labels;
- imbalance setting;
- representation analysis;
- head analysis;
- norm/scale analysis;
- direction/angle analysis;
- endpoint-specific localization;
- endpoint asymmetry;
- sample-level recoverability;
- intervention;
- exact overlap with Candidate C/D.

---

## 11. Novelty decision gate

### GO

Proceed with multi-backbone mechanism replication if the literature position remains:

- representation/classifier decoupling is known;
- classifier rebalancing is known;
- norm and angular effects are known;
- long-tail representation geometry is known;
- ordinal imbalance is known;
- **but endpoint-specific representation/head mismatch and geometry-conditioned recoverability are not already established as the same mechanism.**

### REFRAME

Reframe if close work already establishes direction-specific rare-end recovery in ordinal imbalance but does not examine probability localization or recovery conditions.

Possible fallback:

- center the paper on a descriptive taxonomy of representation-inward versus endpoint-like geometry together with recovery versus non-recovery under the tested direction-only intervention.

Do not label non-recovered samples as representation-limited unless a separate analysis directly establishes that causal limitation.

### NO-GO / major pivot

Do not spend substantial GPU budget if prior work is found that already establishes all of:

- ordinal endpoint inward localization;
- representation/head decomposition;
- direction-specific intervention;
- geometry-conditioned sample recovery.

---

## 12. Final decision

> **GO — bounded mechanism paper claim locked after replication.**

The primary contribution is not that balancing helps a rare class, that
classifier direction matters, or that generic representation/head decoupling
exists.  It is the reproducible ordinal localization response under the
controlled frozen-head intervention across the four RetinaMNIST/Solar
dataset-objective settings.  Endpoint-specific geometry is reported as a
setting-dependent diagnostic rather than a universal explanatory layer.

The completed replication outcome is summarized in
`docs/research/mechanism_cross_setting_synthesis.md`.  No new metric search or
post-hoc H2 alteration is justified by the partial/unsupported Retina results.
