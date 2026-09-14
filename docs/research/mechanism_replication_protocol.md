# Mechanism Replication Protocol

**Status:** **FROZEN FOR EXECUTION** — ICLR 2027 mechanism replication
**Project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Last updated:** 2026-09-13 (Stage 1 provenance audit completed; execution freeze)

The scientific protocol, analysis rules, and execution configuration are now **FROZEN FOR EXECUTION**. Seed 0 remains hypothesis-forming; seeds 1–4 are the confirmatory replication set. The deterministic saved-logit replay has verified that RetinaMNIST RPS seeds 1–4 may be reused under this frozen protocol; see `docs/research/stage1_mechanism_provenance_audit.md`.

---

## 1. Scientific objective

The novelty audit narrows the paper to a specific mechanism question rather than a generic long-tail rebalancing claim.

### Primary scientific question

> **Does direction-only classifier adaptation reproducibly improve rare-end ordinal localization across independently trained backbone representations?**

### Novelty-oriented secondary question

> **Is rare-end recoverability systematically related to how strongly endpoint geometry is retained in the frozen representation?**

These questions are intentionally narrower than:

- whether classifier retraining improves tail accuracy;
- whether balanced sampling is useful;
- whether classifier direction matters in general;
- whether long-tailed training distorts representation geometry.

Those broader facts already have substantial prior art.

---

## 2. Predeclared hypotheses

### H1 — reproducible direction response

For a true rare-end class, changing only classifier directions while keeping the backbone, class-specific weight norms, and biases fixed will improve rare-end localization on average across independently trained backbones.

Primary operational form for seed \(s\):

\[
\Delta^{\mathrm{MAE}}_s
=
\mathrm{MAE}_{C,s}-\mathrm{MAE}_{A,s}.
\]

Improvement corresponds to

\[
\Delta^{\mathrm{MAE}}_s<0.
\]

### H2 — endpoint-specific geometry and recoverability

H2 is split into two distinct questions so that final recoverability is not conflated with intervention change magnitude.

#### H2a — final localization/recovery under C

For a true rare-end sample, stronger endpoint-specific representation geometry should be associated with better final rare-end localization under the direction-only head C, after accounting for how well the original head A already localized that sample.

Primary sample-level outcome:

\[
y_i^{\mathrm{H2a}}
=
(K-1)-\mu_p^{(C)}(x_i),
\]

where smaller values indicate better final localization at the upper endpoint.

Primary representation predictor:

\[
m_{i,\mathrm{adj}}^{\mathrm{rep}}
=
\|h_i-\mu_{K-2}\|_2
-
\|h_i-\mu_{K-1}\|_2.
\]

Larger positive values indicate that the endpoint is closer than the adjacent interior class under this specific endpoint-vs-adjacent diagnostic.

The primary H2a model must also include the original-head predictive location \(\mu_p^{(A)}(x_i)\) so that the analysis does not mistake an already-easy sample for a stronger intervention effect.

#### H2b — intervention response magnitude

Define

\[
\Delta_i^{\mu}
=
\mu_p^{(C)}(x_i)-\mu_p^{(A)}(x_i).
\]

For an upper endpoint, \(\Delta_i^{\mu}>0\) is outward movement toward the endpoint.

H2b asks whether endpoint-specific representation geometry is associated with this A->C change magnitude after conditioning on the original-head state. This is secondary to H2a because strong endpoint geometry can create a ceiling effect when A already performs well.

#### Generic-difficulty comparison

The novelty-oriented question is not merely whether easy samples recover more often. Therefore the H2 analysis must compare endpoint-specific geometry against generic difficulty indicators available under A, including at minimum:

- rare-end probability \(p_{K-1}^{(A)}\);
- rare-vs-adjacent logit margin \(z_{K-1}^{(A)}-z_{K-2}^{(A)}\);
- original predictive location \(\mu_p^{(A)}\);
- a generic centroid-separation measure that does not encode the endpoint-vs-adjacent ordinal relation.

The endpoint-specific geometry term is scientifically informative only if it adds explanatory value beyond these generic difficulty controls.

This H2 specification is the main novelty-oriented hypothesis and must be frozen before inspecting new replication outcomes.

---

## 3. Experimental scope

Primary datasets:

- RetinaMNIST
- Solar flare ordinal classification

Backbone training objectives:

- cross-entropy (CE)
- Ranked Probability Score (RPS)

Primary head comparison:

- **A — original classifier head**
- **C — direction-only adapted classifier head**

Primary settings:

1. RetinaMNIST + CE backbone
2. RetinaMNIST + RPS backbone
3. Solar + CE backbone
4. Solar + RPS backbone

UTKFace remains supporting evidence from the existing study but is not part of the first replication block unless later explicitly authorized.

---

## 4. Unit of replication

The independent replication unit is:

> **an independently trained backbone seed.**

The following do **not** count as independent backbone replications:

- OOF folds from one backbone;
- multiple head seeds on one frozen backbone;
- multiple samples from one evaluation population;
- repeated analysis of the same archived checkpoint;
- multiple decision rules computed from the same model.

This distinction must be preserved in figures, tables, confidence intervals, and manuscript wording.

---

## 5. Predeclared seed set and confirmatory split

Use backbone seeds:

```text
0, 1, 2, 3, 4
```

Target:

- 5 independent backbones per dataset × objective setting;
- 4 settings;
- 20 backbone conditions total.

However, seed 0 has already contributed to hypothesis formation and is therefore **not a new confirmatory replication**. The inferential roles are fixed as follows:

- **confirmatory replication set:** seeds 1, 2, 3, 4;
- **historical/hypothesis-forming seed:** seed 0;
- **all-seed descriptive summary:** seeds 0--4.

Primary replication success criteria in Sections 13 and 17 must be evaluated on seeds 1--4 only. Seed 0 may be shown for continuity and effect estimation, but it must not be counted as a new successful replication.

Existing checkpoints may count only after an exact compatibility audit.

Do not:

- stop early because the first seeds look convincing;
- add seeds only after an unfavorable result;
- remove scientifically poor runs;
- change the seed set after inspecting aggregate outcomes.

---

## 6. Existing-checkpoint compatibility audit

Before training anything, create a checkpoint inventory with one row per candidate backbone:

- dataset;
- backbone objective;
- seed;
- data split identifier;
- architecture;
- input resolution;
- preprocessing;
- augmentation;
- optimizer;
- learning-rate schedule;
- checkpoint-selection rule;
- representation layer used for feature extraction;
- number of classes;
- classifier-head definition;
- evaluation population;
- sample-ID availability;
- existing A outputs;
- existing C outputs;
- compatibility status;
- reason if incompatible.

Allowed compatibility labels:

- `COMPATIBLE`
- `NOT_COMPATIBLE`
- `AMBIGUOUS`

Only exact protocol matches may count toward the five backbone seeds.

**Stage 1 audit result:** 12/20 existing backbone conditions are confirmed compatible. RetinaMNIST CE seeds 0–4 and RPS seeds 0–4 are compatible; Solar CE/RPS seed 0 are compatible. RetinaMNIST RPS seeds 1–4 each uniquely replayed their saved logits with `Normalize((0.5,)*3, (0.5,)*3)`, exact official-test sample IDs and labels, and maximum absolute errors no larger than `3.052e-05`. Solar CE/RPS seeds 1–4 are missing, so eight new backbone trainings are required before the full inventory exists. Seed 0 is descriptive/hypothesis-forming only and does not count toward confirmatory success criteria.

---

## 7. Frozen backbone protocol

For each independent backbone seed:

1. train the backbone using the frozen dataset split and preprocessing;
2. select the checkpoint using the predeclared validation rule;
3. freeze the complete representation network;
4. extract the same penultimate representation used in the existing mechanism study;
5. keep the frozen representation identical for A and C;
6. do not update backbone parameters during the direction-only intervention.

Any change to the following is a protocol change and must be documented before execution:

- architecture;
- stem definition;
- image resolution;
- preprocessing;
- augmentation;
- optimizer;
- scheduler;
- training duration;
- checkpoint-selection rule;
- representation layer;
- train/validation/test split.

The goal is replication, not retuning.

---

## 8. Head parameterization

For class \(k\), write the linear classifier as

\[
z_k=w_k^\top h+b_k,
\qquad
w_k=s_kv_k,
\qquad
s_k=\|w_k\|_2,
\qquad
\|v_k\|_2=1.
\]

### A — original head

Use the original trained:

- direction \(v_k^A\);
- scale \(s_k^A\);
- bias \(b_k^A\).

No head retraining.

### C — direction-only adapted head

Freeze:

- backbone representation;
- original class-specific scales \(s_k^A\);
- original biases \(b_k^A\).

Train only the unit directions \(v_k\):

\[
w_k^C=s_k^A v_k^C.
\]

The adaptation procedure must not silently update scale or bias.

---

## 9. Hard implementation verification for C

Every C run is valid only if the saved parameters satisfy all of the following after adaptation:

### Norm preservation

For every class \(k\):

\[
\big|\|w_k^C\|_2-s_k^A\big|\le \epsilon_{\mathrm{norm}}.
\]

### Bias preservation

For every class \(k\):

\[
|b_k^C-b_k^A|\le \epsilon_{\mathrm{bias}}.
\]

### Backbone preservation

The frozen backbone parameter checksum before and after head adaptation must match exactly or be verified by an equivalent deterministic check.

Numerical tolerances must be set before running the replication.

A violation invalidates the run as a direction-only intervention.

---

## 10. Head-adaptation training protocol

The primary A-vs-C replication must reproduce the canonical direction-only procedure underlying the current Figure 3 evidence.

The canonical C adaptation configuration is now frozen from the audited historical CE/RPS implementations:

- replacement class-balanced sampling;
- cross-entropy head-adaptation objective for both CE- and RPS-trained backbones;
- batch size `64`;
- AdamW optimizer;
- learning rate `1e-3`;
- direction-parameter weight decay `0`;
- fixed `100` epochs;
- initialization from the original A classifier head;
- original per-class row norms fixed;
- original biases fixed;
- only the direction parameter trainable;
- RetinaMNIST uses deterministic five-fold training-only OOF head evaluation on a fixed representation;
- Solar fits C on aligned training features and evaluates on the fixed archived readout population.

Runtime assertions are mandatory: cached/original-head replay and C initialization replay must each have maximum absolute logit error `<=2e-5`; C norm and bias preservation errors must each be `<=1e-6`.

Do not tune these choices using the new replication outcomes.

---

## 11. RetinaMNIST protocol audit

The existing mechanism analysis uses training-only OOF head evaluation on a learned representation.

Before replication, explicitly document:

1. whether the backbone was trained on all training observations;
2. whether only the head adaptation is fold-held-out;
3. how the five folds are constructed;
4. whether fold assignments are deterministic and shared across A/C;
5. whether centroids are computed using only data allowed by the corresponding analysis split;
6. whether held-out head-evaluation observations enter any head-fitting step;
7. whether validation data enter head fitting;
8. whether test data enter any selection step.

If the audit confirms that only the head is OOF, use precise manuscript wording such as:

> **training-only out-of-fold head evaluation on a fixed learned representation**

Do not describe it as fully OOF backbone evaluation.

**Frozen result:** the backbone is trained before OOF; only head fitting/evaluation is five-fold OOF. Historical all-centroid subgroups used full-training-feature centroids and remain descriptive only. Confirmatory H2 geometry must instead use fitting-fold-only centroids for each held RetinaMNIST fold, then concatenate held-fold geometry values.

---

## 12. Solar protocol audit

Before replication, explicitly document:

- chronological train/validation/test boundaries;
- exact evaluation/readout population;
- sample identifiers;
- label derivation;
- timestamp matching/alignment;
- duplicate handling;
- whether the archived readout has informed prior exploratory interpretation;
- whether all seeds use the identical evaluation population;
- whether any model-selection decision uses the readout population.

The existing Solar readout must not be described as a newly independent confirmatory test if it has already been reused for mechanism analysis.

The replication can still test **backbone-realization robustness** on the same fixed readout population, but the manuscript must distinguish this from a new external validation set.

**Frozen result:** the Solar evaluation population is the same archived aligned readout used previously (approximately 2020–2024 test; train/validation primarily 2010–2019). It is reused for backbone-realization robustness and must not be described as a new independent external confirmation. Centroids for H2 are computed from aligned training features only; no validation/test sample contributes to centroid construction or model selection.

---

## 13. Primary endpoint and H1 success criterion

Primary endpoint:

> **rare-end class-conditional L1 MAE under the exact L1 Bayes decision.**

For seed \(s\):

\[
\Delta^{\mathrm{MAE}}_s
=
\mathrm{MAE}_{C,s}-\mathrm{MAE}_{A,s}.
\]

Improvement corresponds to

\[
\Delta^{\mathrm{MAE}}_s<0.
\]

### 13.1 Confirmatory H1 criterion

For each dataset × backbone-objective setting, H1 is judged using **new seeds 1--4 only**:

- **replicated in the setting:** all 4 of 4 new seeds satisfy \(\Delta^{\mathrm{MAE}}_s<0\);
- **partial replication in the setting:** exactly 3 of 4 new seeds satisfy \(\Delta^{\mathrm{MAE}}_s<0\);
- **not replicated in the setting:** 2 or fewer of 4 new seeds satisfy \(\Delta^{\mathrm{MAE}}_s<0\).

This sign-consistency rule is the primary replication gate. With only four new independent backbones per setting, a confidence interval is reported for uncertainty but is **not** required to exclude zero for the setting to satisfy the sign-based replication criterion.

### 13.2 H1 reporting

For each setting, report separately:

**Confirmatory replication summary (seeds 1--4):**

- all 4 paired seed effects;
- mean paired effect;
- standard deviation of paired effects;
- two-sided 95% Student-\(t\) confidence interval for the mean paired effect using the 4 seed-level effects;
- number of new seeds with \(\Delta^{\mathrm{MAE}}_s<0\);
- replication category from Section 13.1.

**All-seed descriptive summary (seeds 0--4):**

- all 5 paired effects;
- mean and standard deviation;
- two-sided 95% Student-\(t\) confidence interval using the 5 seed-level effects.

The all-seed summary must be labeled descriptive because seed 0 was used during hypothesis formation.

Do not pool test samples across backbone seeds and use the sample count as the inferential replication count. Emphasize effect magnitude and seed-level sign consistency over a standalone p-value.

---

## 14. Secondary localization endpoints

For each backbone seed, report at minimum:

### 14.1 Exact rare-end recovery

- fraction;
- raw numerator/denominator.

Use the same exact L1 decision rule as the primary endpoint.

### 14.2 Inward shrinkage

For upper endpoint \(K-1\):

\[
\mathrm{shrinkage}
=
(K-1)-\frac{1}{n_{\mathrm{end}}}
\sum_{i:Y_i=K-1}\mu_p(x_i).
\]

### 14.3 Rare-vs-adjacent logit margin

\[
m_i^{\mathrm{logit}}
=
z_{K-1}(x_i)-z_{K-2}(x_i).
\]

Report the class-conditional mean and paired A->C change.

### 14.4 Rare-end probability

\[
p_{K-1}(x_i).
\]

Report the class-conditional mean and paired A->C change.

---

## 15. Collateral-effect reporting

Rare-end recovery must not be reported without side effects.

For every A/C pair, report at minimum:

- global L1 MAE;
- lower-endpoint L1 MAE;
- overall exact accuracy if already part of the pipeline;
- RPS or the canonical probability-quality metric already used in the study;
- any predeclared severe-error metric if available.

The paper must distinguish:

> local rare-end localization recovery

from

> globally better classification or probability quality.

Direction-only adaptation is a mechanistic probe, not automatically a better classifier.

---

## 16. Representation geometry definition

Centroids must be derived from training-allowed observations only.

For class \(k\):

\[
\mu_k
=
\frac{1}{N_k}\sum_{i:y_i=k} h(x_i).
\]

For each true rare-end evaluation sample \(i\), define:

\[
d_{i,\mathrm{end}}
=
\|h_i-\mu_{K-1}\|_2,
\]

\[
d_{i,\mathrm{adj}}
=
\|h_i-\mu_{K-2}\|_2,
\]

and the predeclared continuous representation margin:

\[
m_i^{\mathrm{rep}}
=
d_{i,\mathrm{adj}}-d_{i,\mathrm{end}}.
\]

Interpretation:

- \(m_i^{\mathrm{rep}}>0\): closer to the rare-end centroid than the adjacent centroid;
- \(m_i^{\mathrm{rep}}<0\): closer to the adjacent interior centroid;
- magnitude measures relative endpoint alignment under this centroid diagnostic.

Important: this endpoint-vs-adjacent margin is **not** the same as the historical binary subgroup used in the current manuscript figures.

Preserve the historical binary subgroup exactly:

- `rare-end-like`: the rare-end centroid is the nearest centroid among **all** class centroids;
- `representation-inward`: some non-endpoint class centroid is nearest.

Formally, for true rare-end sample \(i\),

\[
\arg\min_{k\in\{0,\ldots,K-1\}}
\|h_i-\mu_k\|_2
=
K-1
\]

defines `rare-end-like`; otherwise the sample is `representation-inward`.

The continuous quantity \(m_{i,\mathrm{adj}}^{\mathrm{rep}}\) must be named **endpoint-vs-adjacent representation margin** and analyzed separately. A sample can have a positive endpoint-vs-adjacent margin while still being closer to another interior centroid than to the endpoint.

Neither diagnostic is a causal proof of representation quality, limitation, or irrecoverability.

---

## 17. Endpoint-geometry and recoverability analysis

This analysis is predeclared because it is the strongest remaining novelty hypothesis after the literature audit. H2a is the sole primary geometry analysis. H2b is an alternative change-score parameterization of closely related information and must not be counted as independent replication evidence.

### 17.1 Historical binary subgroup

Retain the existing all-centroid subgroup only as a descriptive continuity analysis:

- endpoint centroid nearest among all classes;
- another class centroid nearest.

Report recovery and non-recovery under C separately for these groups, but do not infer causal representation limitation from non-recovery.

### 17.2 Fixed primary H2a outcome

For each true rare-end sample \(i\), define final inward shrinkage under C:

\[
Y_i^{(C)}=(K-1)-\mu_p^{(C)}(x_i).
\]

Smaller values indicate better final localization at the upper endpoint.

Define the corresponding original-head state:

\[
Y_i^{(A)}=(K-1)-\mu_p^{(A)}(x_i).
\]

The primary endpoint-specific representation predictor is:

\[
m_{i,\mathrm{adj}}^{\mathrm{rep}}
=
\|h_i-\mu_{K-2}\|_2-\|h_i-\mu_{K-1}\|_2.
\]

Larger values indicate stronger endpoint-versus-adjacent alignment under this diagnostic. The expected beneficial coefficient direction is **negative** because stronger endpoint alignment should correspond to smaller final shrinkage.

### 17.3 Generic geometry control

Define a generic, order-agnostic centroid-separation margin from all class-centroid distances. Let \(d_{i,(1)}\) and \(d_{i,(2)}\) denote the smallest and second-smallest distances among \(\{\|h_i-\mu_k\|_2\}_{k=0}^{K-1}\). Define:

\[
g_i^{\mathrm{centroid}}=d_{i,(2)}-d_{i,(1)}.
\]

Larger values indicate a more clearly separated nearest centroid regardless of which class is nearest. This variable deliberately does not encode the endpoint-versus-adjacent ordinal relation.

The definition of \(g_i^{\mathrm{centroid}}\) is fixed before new outcomes are inspected and must not be replaced post hoc by another generic margin because it produces more favorable results.

### 17.4 Fixed nested-model test for H2a

The primary H2a analysis compares two fixed ordinary least-squares models within each backbone seed. Continuous predictors are standardized to zero mean and unit standard deviation **within that backbone's rare-end evaluation population** before fitting. The outcome remains in its original shrinkage units.

Base model:

\[
M_0:\quad
Y_i^{(C)}
=
\beta_0+\beta_A Y_i^{(A)}+\beta_g g_i^{\mathrm{centroid}}+\epsilon_i.
\]

Ordinal endpoint model:

\[
M_1:\quad
Y_i^{(C)}
=
\beta_0+\beta_A Y_i^{(A)}+\beta_g g_i^{\mathrm{centroid}}
+\beta_{\mathrm{ord}}m_{i,\mathrm{adj}}^{\mathrm{rep}}+\epsilon_i.
\]

The primary geometry effect is \(\beta_{\mathrm{ord}}\). The predeclared beneficial direction is:

\[
\beta_{\mathrm{ord}}<0.
\]

This nested comparison operationalizes the novelty question: does endpoint-specific ordinal geometry add information beyond the original-head localization state and a generic order-agnostic class-separation measure?

### 17.5 Confidence intervals and incremental explanatory value

For each backbone seed:

1. fit \(M_0\) and \(M_1\) on the full rare-end evaluation population;
2. report the standardized-predictor coefficient \(\hat\beta_{\mathrm{ord}}\);
3. report a two-sided 95% HC3 heteroskedasticity-robust confidence interval for \(\hat\beta_{\mathrm{ord}}\);
4. compute leave-one-out cross-validated mean squared error for both models;
5. define incremental predictive value as

\[
\Delta_{\mathrm{LOO}}
=
\mathrm{MSE}_{\mathrm{LOO}}(M_0)-\mathrm{MSE}_{\mathrm{LOO}}(M_1).
\]

Positive \(\Delta_{\mathrm{LOO}}\) means that adding endpoint-specific geometry improves out-of-sample prediction of final shrinkage relative to the fixed base model. Standardization used during LOOCV must be estimated on each training fold and then applied to the held-out sample.

HC3 intervals are sample-level diagnostics within a backbone; they do not change the independent replication unit from backbone seed to sample.

### 17.6 Collinearity and alternative A-state controls

The primary model intentionally uses only \(Y^{(A)}\) as the original-head state control rather than simultaneously entering \(p_{K-1}^{(A)}\), \(z_{K-1}^{(A)}-z_{K-2}^{(A)}\), and \(\mu_p^{(A)}\). These quantities are expected to be correlated, and RetinaMNIST has only 66 rare-end samples in the historical setting.

Before inspecting new C outcomes, Stage 1 must compute predictor-only correlations among:

- \(Y^{(A)}\);
- \(p_{K-1}^{(A)}\);
- \(z_{K-1}^{(A)}-z_{K-2}^{(A)}\);
- \(g^{\mathrm{centroid}}\);
- \(m_{\mathrm{adj}}^{\mathrm{rep}}\).

These correlations are diagnostic only and do not change the primary \(M_0/M_1\) specification.

Sensitivity analyses may replace \(Y^{(A)}\) one at a time with:

- rare-end probability \(p_{K-1}^{(A)}\); or
- rare-vs-adjacent logit margin \(z_{K-1}^{(A)}-z_{K-2}^{(A)}\).

Do not place all correlated A-state controls into one primary model, and do not select the sensitivity control based on which one makes the endpoint-specific coefficient most favorable. Sensitivity analyses cannot override the primary \(M_0/M_1\) result.

### 17.7 Confirmatory H2a success criterion

H2a is judged using **new seeds 1--4 only** in each dataset × objective setting.

A setting provides **strong H2a replication** only if both conditions hold:

1. \(\hat\beta_{\mathrm{ord}}<0\) in all 4 of 4 new backbone seeds; and
2. \(\Delta_{\mathrm{LOO}}>0\) in at least 3 of the 4 new backbone seeds, with the mean \(\Delta_{\mathrm{LOO}}\) across seeds 1--4 also positive.

A setting provides **partial H2a replication** if:

- the beneficial coefficient sign occurs in exactly 3 of 4 new seeds, or
- the coefficient sign is 4 of 4 beneficial but the incremental-LOO criterion above is not met.

H2a is **not replicated** in the setting if the beneficial coefficient sign occurs in 2 or fewer of the 4 new seeds.

For uncertainty reporting, summarize \(\hat\beta_{\mathrm{ord}}\) across seeds 1--4 using the mean, standard deviation, and a two-sided 95% Student-\(t\) confidence interval over the four seed-level coefficient estimates. Report the analogous seeds 0--4 summary separately as descriptive. The cross-seed confidence interval is not an additional binary success gate because \(n=4\) confirmatory backbones is small.

Seed 0 must never be used to satisfy the confirmatory sign or incremental-LOO criteria.

### 17.8 Secondary H2b change-score analysis

For each sample, define:

\[
\Delta_i^{\mu}
=
\mu_p^{(C)}(x_i)-\mu_p^{(A)}(x_i),
\]

\[
\Delta_i^{p}
=
p_{K-1}^{(C)}(x_i)-p_{K-1}^{(A)}(x_i).
\]

These change-score analyses may be reported to aid interpretation of the intervention. However, after conditioning on the A-state quantity, final-C localization and C-minus-A change are algebraically linked parameterizations of closely related information. Therefore:

- H2b has no separate replication-success criterion;
- agreement between H2a and H2b is not counted as independent confirmation;
- H2b cannot rescue a failed H2a primary analysis.

### 17.9 Exact recovery

Exact recovery under C is a secondary binary endpoint.

For samples not exactly recovered by A, define:

\[
R_i
=
\mathbf{1}\{\hat y^{(C)}_{L1}=K-1\}.
\]

If A already recovers any sample exactly, report transition categories explicitly rather than silently treating every C endpoint prediction as a recovery.

Use exact-recovery rates and the historical all-centroid subgroup for descriptive support. Do not use exact recovery as the sole H2 criterion because RetinaMNIST has a small rare-end denominator and the outcome is discrete.

---

## 18. Statistical reporting principles

The inferential hierarchy is:

1. independent backbone seed;
2. paired A/C effect within backbone;
3. sample-level analyses nested within backbone.

Primary H1 evidence:

- confirmatory paired effects from seeds 1--4;
- 4-of-4 / 3-of-4 / <=2-of-4 sign category from Section 13;
- effect magnitude and Student-\(t\) uncertainty across backbone seeds;
- seeds 0--4 shown separately as descriptive continuity.

Primary H2 evidence:

- per-backbone \(\hat\beta_{\mathrm{ord}}\) from the fixed \(M_1\) model;
- confirmatory sign consistency across seeds 1--4;
- predeclared incremental LOOCV criterion relative to \(M_0\);
- cross-seed mean/SD/Student-\(t\) interval for \(\hat\beta_{\mathrm{ord}}\);
- seeds 0--4 shown separately as descriptive continuity.

Secondary H2 evidence:

- HC3 within-backbone intervals;
- H2b change-score analyses;
- exact-recovery summaries;
- historical all-centroid subgroup summaries;
- one-at-a-time A-state sensitivity controls.

Do not treat H2a and H2b as independent pieces of replication evidence. Do not let the number of rare-end evaluation samples, especially in Solar, replace the number of independently trained backbones as the replication count.

---

## 19. Factorial follow-up: sampling × objective

This is **secondary** to H1/H2.

Run only after the primary A/C replication pipeline is validated and the backbone replication is complete or clearly on track.

On frozen RetinaMNIST RPS representations, repeat the four direction-only cells:

- Natural + CE
- Balanced + CE
- Natural + RPS
- Balanced + RPS

for each compatible independent backbone seed.

Primary factorial outcomes:

- class-4 L1 MAE;
- mean \(z_4-z_3\) margin.

Purpose:

> Test whether the current pattern `Balanced CE ≈ Balanced RPS < Natural CE ≈ Natural RPS` is stable across independently trained representations.

Do not present balanced sampling as a new method.

If compute is limited, this factorial replication is lower priority than H1/H2.

Priority ordering:

\[
\boxed{
\text{A/C backbone replication}
>
\text{geometry-conditioned recoverability}
>
\text{sampling × objective replication}
}
\]

---

## 20. Scale and bias controls

Multi-seed scale/bias controls are **not required** for the first replication block.

Claim rule:

- if only A/C is replicated, use:
  > **direction-only adaptation provides a reproducible head-level localization response**;

- use the stronger phrase
  > **direction is the most consistent actionable head component**

  only if scale/bias alternatives are also replicated sufficiently to justify the comparison.

Do not let the manuscript claim outrun the replicated controls.

---

## 21. Exclusion rules

A run may be excluded only for a predeclared technical failure, such as:

- corrupted or unreadable checkpoint;
- incomplete training caused by job failure;
- data/sample-ID misalignment;
- missing required predictions;
- violation of frozen backbone constraint;
- violation of fixed norm/bias constraint;
- deterministic code failure that invalidates outputs.

Poor scientific performance is **not** an exclusion criterion.

Every exclusion must be logged before aggregate interpretation.

If a failed run is rerun, preserve the failure record and use the same seed/configuration.

---

## 22. Required artifacts per backbone seed

Save enough information to reconstruct every table/figure without retraining.

Required:

- dataset/config identifier;
- backbone objective;
- backbone seed;
- checkpoint path/hash or immutable identifier;
- train/validation/test split identifiers;
- original head parameters;
- C head parameters;
- per-class A norms;
- per-class C norms;
- per-class A biases;
- per-class C biases;
- backbone-freeze verification;
- rare-end sample IDs;
- true labels;
- A logits;
- C logits;
- A probabilities;
- C probabilities;
- A L1 decisions;
- C L1 decisions;
- predictive means;
- rare-end probabilities;
- adjacent logit margins;
- feature vectors or reproducible feature references;
- centroid definitions/provenance;
- continuous representation margins;
- primary/secondary metrics;
- software/config provenance.

The manuscript-data pipeline should aggregate from these saved artifacts, not from hand-entered summary values.

---

## 23. Execution sequence

### Stage 0 — freeze novelty target

- [x] Initial novelty audit completed.
- [x] Central novelty narrowed to endpoint representation/head mismatch and recoverability.
- [x] Targeted novelty-positioning audit completed through the current literature review; any later-discovered direct overlap must be documented without changing outcomes post hoc.

### Stage 1 — protocol/provenance audit

- [x] Recover exact canonical A/C implementation details.
- [x] Audit RetinaMNIST OOF semantics.
- [x] Audit Solar evaluation/readout provenance.
- [x] Build checkpoint compatibility inventory (12/20 compatible; Retina RPS seeds 1–4 replay-verified).
- [x] Set numerical tolerances for replay/norm/bias checks.
- [x] Freeze the historical all-centroid subgroup definition separately from the endpoint-vs-adjacent margin.
- [x] Freeze the H2a regression/association model and confidence-interval method.
- [x] Freeze the generic centroid-separation control definition.
- [x] Record seed 0 as hypothesis-forming and seeds 1--4 as the new replication subset.
- [x] Freeze all configs and analysis definitions.

### Stage 2 — RetinaMNIST replication

- [ ] Validate the full artifact pipeline on one compatible seed without changing the protocol.
- [ ] Complete CE backbone seeds 0–4.
- [ ] Complete RPS backbone seeds 0–4.
- [ ] Verify all A/C constraints.
- [ ] Run predeclared H1/H2 analysis.

### Stage 3 — Solar replication

- [ ] Complete CE backbone seeds 0–4.
- [ ] Complete RPS backbone seeds 0–4.
- [ ] Verify all A/C constraints.
- [ ] Run the same predeclared H1/H2 analysis.

### Stage 4 — optional factorial replication

- [ ] Decide based on predeclared scientific need/compute budget, not based on cherry-picking favorable H1/H2 outcomes.
- [ ] If run, repeat the RetinaMNIST RPS 2×2 direction-only factorial across independent backbones.

### Stage 5 — claim lock

- [ ] Compare final outcomes against decision rules below.
- [ ] Freeze manuscript claim strength before revising figures/results.

---

## 24. Decision rules after replication

### Strong replication

Conditions:

- H1 is **replicated** by the Section 13 criterion (4/4 new seeds improve) in every dataset × objective setting included in the claimed cross-setting mechanism statement;
- secondary localization measures are directionally compatible with the H1 interpretation and major collateral trade-offs are reported;
- H2a achieves **strong H2a replication** by Section 17.7 in every setting included in any claim that endpoint-specific geometry adds information beyond generic difficulty.

Allowed central claim:

> Rare-end samples can retain endpoint-aligned geometry while being localized inward by the classifier head. Direction-only adaptation produces a reproducible head-level localization response, and in the settings where H2a strongly replicates, endpoint-specific representation geometry adds predictive information about final localization beyond the original-head state and generic centroid separation.

### Moderate replication

Conditions:

- H1 replicates strongly in some settings and partially in others; or
- H1 is broadly stable but H2a strongly replicates only in a subset of datasets/objectives.

Allowed claim:

> Direction-only adaptation provides a reproducible but conditional head-level localization response. Endpoint-specific representation geometry provides additional information about final localization only in the settings where the predeclared H2a criterion is met.

Do not claim a universal recoverability law.

### Partial replication

Conditions:

- H1 is only partial by Section 13 in one or more key settings; or
- effect sign varies across dataset/objective settings such that a cross-domain statement is not supported.

Action:

- narrow the paper to the replicated conditions;
- treat cross-domain results as heterogeneous;
- do not use "reproducible across objectives/domains" broadly;
- keep H2 as descriptive unless its own predeclared criterion is met in a clearly delimited setting.

### Weak / failed replication

Conditions:

- H1 is not replicated (2/4 or fewer new seeds improve) in the central settings;
- apparent A/C recovery is dominated by the historical seed 0 or a single new backbone;
- H2a is not replicated under the fixed primary model and confirmatory seeds.

Action:

- remove direction/recoverability as the central mechanism claim;
- reframe around descriptive endpoint localization and conditional failure taxonomy, or reconsider the ICLR submission direction.

---

## 25. Claim boundaries after this protocol

Even under strong replication, do not claim:

- imbalance is the isolated cause of inward localization;
- balanced sampling is a new method;
- direction is a novel concept in long-tail learning;
- endpoint-aligned centroid geometry proves causal recoverability;
- representation-inward samples are irrecoverable;
- C is globally superior to A;
- CE and RPS are equivalent;
- the mechanism is universal across ordinal tasks.

The intended contribution remains a bounded empirical mechanism characterization.

---

## 26. Current authorization status

> **FROZEN FOR EXECUTION.**

The experimental design, H1/H2 definitions, confirmatory seed roles, head-adaptation protocol, centroid rules, generic-difficulty control, success criteria, and numerical integrity tolerances are frozen before new replication outcomes are inspected.

Execution authorization:

1. **Solar CE/RPS seeds 1–4:** authorized for backbone training under the frozen Phase 3.8-compatible protocol. This is eight new backbone trainings.
2. **RetinaMNIST CE seeds 1–4:** compatible existing checkpoints may be reused; no retraining is authorized unless a later integrity failure is documented.
3. **RetinaMNIST RPS seeds 1–4:** deterministic saved-logit replay against the canonical local RetinaMNIST split is complete. Reuse the verified checkpoints with `Normalize((0.5,)*3, (0.5,)*3)`; do not alter H1/H2 rules.
4. **Seed 0:** historical/hypothesis-forming only; it may appear in all-seed descriptive summaries but is excluded from confirmatory success decisions.
5. **A/C replication:** authorized once a backbone condition is verified compatible or newly trained under this frozen protocol.

No method redesign, scale/bias replication, Phase 3.19 factorial replication, dataset expansion, threshold tuning, or success-criterion change is authorized by this freeze. Any technical correction required to execute the frozen design must be documented without using scientific outcomes to choose the correction.
