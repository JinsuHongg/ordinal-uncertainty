# Experiment Plan

## Current Stage
Completed through **Phase 3.1**.

Current gate:

\[
\boxed{\text{PHASE 3.2 JUSTIFIED}}
\]

The next experiment should test whether a minimal objective can reduce rare upper-extreme inward shrinkage while preserving the useful decision-risk geometry observed with RPS.

No proposed method is frozen yet.

## Canonical Experimental Setup
### Dataset
Primary development dataset: **RetinaMNIST**

Use:
- native **28×28 RGB**
- official train/validation/test splits
- fixed class order `0,1,2,3,4`

Class counts:

| Split | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---:|---:|---:|---:|---:|
| Train | 486 | 128 | 206 | 194 | 66 |
| Validation | 54 | 12 | 28 | 20 | 6 |
| Test | 174 | 46 | 92 | 68 | 20 |

Class 4 is the rare upper extreme.

### Backbone
- unpretrained ResNet18
- 3×3 stride-1 first convolution
- no max-pool
- same backbone across probability-based baselines unless a published method requires otherwise

### Seeds
Primary seeds: `0, 1, 2, 3, 4`

For a new Phase 3.2 candidate:
1. unit tests
2. short seed-0 smoke run
3. full seed-0 GO/NO-GO
4. seeds 1–4 only after explicit review

Do not launch multi-seed training automatically.

## Core Evaluation Decomposition
Evaluate probability-based methods as:

\[
\text{Predictive distribution}
\rightarrow
\text{Decision rule}
\rightarrow
\text{Expected decision risk}.
\]

### Decision rules
Mode:

\[
\hat y_{\mathrm{mode}}=\arg\max_k p_k.
\]

L1-optimal:

\[
\hat y_{L1}
=
\arg\min_a\sum_kp_k|k-a|.
\]

L2-optimal:

\[
\hat y_{L2}
=
\arg\min_a\sum_kp_k(k-a)^2.
\]

Use the existing exact discrete implementations and established tie-breaking rule.

### Decision risks
Mode-centered L1 risk:

\[
R_{L1}^{\mathrm{mode}}
=
\sum_kp_k|k-\hat y_{\mathrm{mode}}|.
\]

L1 Bayes risk:

\[
R_{L1}^{*}
=
\min_a\sum_kp_k|k-a|.
\]

L2 Bayes risk:

\[
R_{L2}^{*}
=
\min_a\sum_kp_k(k-a)^2.
\]

These are established decision-risk quantities, not proposed new uncertainty measures.

## Completed Stages

### Experiment 0 / Phase 1
Status: **COMPLETE**

Established the native-28 CE baseline and showed that simple ordinal uncertainty measures do not universally dominate nominal uncertainty.

### Phase 1.5
Status: **COMPLETE**

Strong ordinal-UQ audit showed that decision-risk quantities are more useful than inventing a new simple ordinal spread metric.

### Resolution Sanity Check
Status: **COMPLETE**

Decision: native **28×28** is canonical. Historical 64×64 results remain sensitivity evidence only.

### Phase 1.75
Status: **COMPLETE**

L1-optimal decisions improve MAE and severe-error burden relative to mode. L2 can reduce severe errors further but with a larger accuracy trade-off.

Critical control:

\[
\boxed{\text{Compare new ordinal models against CE + L1}}
\]

not only CE + mode.

### Phase 2
Status: **COMPLETE**

Methods:
- CE
- CORAL
- RPS

Decisions:
- CORAL: **STOP — SCIENTIFICALLY NONCOMPETITIVE**
- RPS: **RETAIN**

RPS gives a mixed model-level signal: stronger risk alignment, severe-error detection, and ordinal selective prediction, but no uniform ordinal-decision improvement.

### Phase 2.5
Status: **COMPLETE**

Validation-only temperature scaling does not remove the RPS decision-risk advantage.

Decision: **MIXED AFTER CALIBRATION**.

### Phase 3.0
Status: **COMPLETE**

Rare class 4 shows:
- suppressed true-class probability
- strong center shrinkage
- high risk but poor location
- no recovery from L1/L2 decision changes
- no correction from temperature scaling

Primary actionable failure: **ordinal center shrinkage under imbalance**.

### Phase 3.1
Status: **COMPLETE**

Baselines:
- Weighted CE
- SLACE

Weighted CE:
- increases class-4 mass somewhat
- does not recover class-4 decisions
- damages global performance
- **STOP**

SLACE:
- does not reduce class-4 shrinkage relative to CE
- no exact or meaningful adjacent class-4 recovery
- retains some risk-ranking utility
- **STOP**

Decision:

\[
\boxed{\text{SHRINKAGE PERSISTS AFTER STRONG BASELINES}}
\]

## Phase 3.2 — Current Stage

Candidate-method design is justified but has **not started**. No proposed
objective is frozen.

### Research Question
> Can a minimal training objective reduce rare upper-extreme inward shrinkage while preserving RPS-like ordinal risk alignment, severe-error detection, selective prediction, and global predictive quality?

### Required Contribution Gap
A Phase 3.2 candidate is only justified if it targets a failure not adequately resolved by:
- CE
- RPS
- class-weighted CE
- SLACE
- scalar temperature scaling
- L1/L2 Bayes decision correction

### Candidate-Design Principles
Do not freeze a method before literature review and diagnostic reasoning.

Candidate directions may include:
- probability-location correction
- class-conditional target-distribution shaping
- ordinal location-bias regularization
- extreme-aware cumulative-distribution penalties
- a minimal RPS-based location correction

These are hypotheses only.

### Seed-0 GO/NO-GO Protocol
For each candidate:
1. verify overlap with existing literature
2. state the exact failure mechanism it targets
3. implement tests
4. run a short seed-0 smoke test
5. run one full seed-0 experiment
6. compare with frozen CE and RPS seed-0 baselines
7. inspect class-4 geometry before any multi-seed expansion

## Primary Global Metrics
Report:
- Accuracy
- MAE
- QWK
- severe-error prevalence
- NLL
- Brier score
- RPS
- ECE

## Primary Decision-Controlled Metrics
For mode, L1, and L2:
- Accuracy
- MAE
- QWK
- severe-error prevalence

Primary global control:

\[
\boxed{\text{RPS + L1}}
\]

A candidate is not successful merely because it beats CE + mode.

## Primary Class-4 Metrics
For true class 4:
- Accuracy
- MAE
- severe-error prevalence
- `4→4`, `4→3`, `4→2`, `4→1`, `4→0`
- mean `p4`
- mean `p3+p4`
- predictive mean
- inward shrinkage
- decision bias
- L1 Bayes risk

The main question is whether probability mass and decisions actually move toward the true upper extreme.

## Class-0 Control
For true class 0:
- Accuracy
- MAE
- severe-error prevalence
- `p0`
- `p0+p1`
- predictive mean
- inward bias/shrinkage
- L1 Bayes risk

Any class-4 gain must be checked for damage to the majority lower extreme.

## Risk-Quality Metrics
Using L1-optimal decisions and L1 Bayes risk:
- risk/error Spearman
- severe-error AUROC
- severe-error AUPRC
- ordinal-MAE risk–coverage
- mean selective MAE risk

A successful candidate should preserve or improve the useful risk geometry seen with RPS.

## Phase 3.2 Decision Rules

### GO
Proceed to multi-seed validation if a candidate meaningfully improves rare upper-extreme geometry while maintaining acceptable global and risk quality.

Possible evidence:
- higher class-4 `p4`
- higher `p3+p4`
- predictive mean closer to 4
- lower inward shrinkage
- fewer `4→0/1/2` errors
- more `4→3/4` outcomes
- lower class-4 MAE/severe burden
- acceptable overall accuracy/QWK
- preserved RPS-like risk alignment and selective prediction

### TRADE-OFF
Pause before expansion if class-4 recovery causes substantial:
- overall Accuracy/QWK loss
- class-0 collapse
- worse severe-error detection
- worse selective prediction

Document the exact trade-off first.

### NO-GO
Stop a candidate if it:
- does not improve class-4 location geometry
- only changes confidence/sharpness
- reproduces generic class weighting
- severely damages global performance
- destroys useful risk alignment

Preserve valid negative results.

## Artifact Requirements
Every full run must save:
- `config.json`
- training history
- validation history
- selected epoch / validation criterion
- `best_checkpoint.pt`
- logits/probabilities
- labels/sample IDs
- predictive metrics
- decision metrics
- classwise metrics
- risk-alignment metrics
- severe-detection metrics
- risk–coverage artifacts
- extreme-class/shrinkage diagnostics

## Future Multi-Seed Gate
Only after seed-0 GO:
- run seeds 1–4
- aggregate mean ± SD
- inspect paired per-seed differences
- confirm class-4 gains are not seed-specific
- confirm stable risk-quality preservation

## Multi-Dataset Expansion
Do not begin the full suite until a Phase 3.2 candidate survives RetinaMNIST.

Potential later datasets:
- UTKFace
- solar flare ordinal classification
- Amazon Reviews
- additional ordinal benchmarks from the literature

## Phase 4 — Epistemic UQ
Not active.

Only after the single-model method is established:
1. MC Dropout
2. Deep Ensemble with `M=3`
3. larger/snapshot ensemble only if justified

The epistemic backend is not the primary novelty target.

## Current Guardrails
Do not currently:
- add a new uncertainty metric
- restart CORAL
- rerun Weighted CE or SLACE multi-seed
- return to 64×64 RetinaMNIST
- add ensembles
- add conformal prediction
- tune on test data
- launch all datasets
- claim novelty before literature verification and seed-0 evidence

## Next Authorized Work
When a separate Phase 3.2 task authorizes candidate design, begin with a small set
of minimal objectives that explicitly target **rare upper-extreme
probability-location shrinkage**.

Before training:
1. check overlap with existing literature
2. define each candidate's intended mechanism
3. specify seed-0 GO/NO-GO criteria
4. implement only the smallest candidate needed for diagnosis

Do not launch multi-seed training until one candidate shows a clear seed-0 signal.
