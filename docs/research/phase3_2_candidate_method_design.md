# Phase 3.2 — Candidate-Method Design for Rare Upper-Extreme Shrinkage

## Scope and status

This is a design and literature-overlap audit only.  No Phase 3.2 loss,
trainer, test, smoke run, model, checkpoint, or output artifact has been
implemented or launched.  The canonical setting remains native 28x28 RGB
RetinaMNIST with the official split and small-image ResNet18.

The aim is deliberately narrow:

> Reduce rare-endpoint inward probability-location shrinkage while retaining
> RPS-like ordinal risk alignment, severe-error detection, selective
> prediction, and global predictive quality.

The proposed names below are working descriptions, not novelty claims.

## Established failure mechanism

The observed problem is **ordinal location shrinkage**, with class imbalance as
a plausible contributor but not a sufficient explanation.  On true class 4,
CE and RPS place their mean probability vectors around the centre: raw RPS has
mean `p4=0.036`, `p3+p4=0.343`, predictive mean `1.744` (truth 4), and inward
shrinkage `1.595` in the five-seed Phase 3.0 audit.  In the matched Phase 3.1
seed-0 comparison, RPS likewise has `p4=0.0837`, `p3+p4=0.3524`, predictive
mean `1.9133`, and no class-4 L1 decision recovery.

This is not adequately characterized as:

* **generic confidence/calibration failure:** validation-only scalar
  temperature scaling improves NLL/ECE but does not move class-4 mass outward;
  it slightly worsens RPS class-4 shrinkage and severe prevalence;
* **decision-rule failure:** mode, exact discrete L1, and exact discrete L2
  decisions all fail to predict class 4; and
* **generic class-frequency failure:** inverse-frequency weighted CE increases
  `p4` but has 0% class-4 L1 accuracy, worse class-4 MAE, and substantially
  worse global quality.

The central diagnostic is instead: for a rare upper endpoint, the model often
reports high L1 Bayes risk but concentrates probability at central/adjacent
labels, so the distribution recognizes unresolved difficulty without locating
the upper tail.  Class 0 is a required control, because it is much better
localized and the formulation must nevertheless apply symmetrically to both
endpoints.

## Design requirements

1. Retain Softmax class probabilities and the RPS base signal.
2. Target probability *location*, not a post-hoc action or generic sharpness.
3. Define endpoints generically as `y in {0, K-1}`; never special-case class 4.
4. If frequency enters, derive it solely from training counts and keep it
   separable from the ordinal-location component.
5. Do not replace the held-out validation role with class-4-test optimization.
6. Preserve a complete predictive distribution rather than optimize only a
   scalar decision.

## Focused literature-overlap audit

The search covered ordinal imbalance/cost sensitivity, RPS/CRPS and
Wasserstein/EMD losses, soft and asymmetric ordinal targets, label-distribution
learning, long-tailed ordinal classification, and extreme/central-tendency
bias.  The resulting implications are deliberately conservative.

| Area | Closest relevant work and implication |
| --- | --- |
| RPS/CRPS ordinal probability scoring | The RPS is a proper CDF scoring rule; it is already the project anchor.  Recent work also connects ordinal scoring/decisions to label-dependent losses.  A threshold-reweighted RPS is therefore an adaptation, not an unclaimed new family. [Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437); [Delgado, 2026](https://doi.org/10.1007/s10994-026-07023-z) |
| Mean supervision | Mean--variance distribution learning directly penalizes the predictive mean's distance from the target, so a mean-only term has clear published precedent. [Pan et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.html) |
| Soft/structured ordinal targets | Metric soft labels, triangular/generalized-triangular labels, and ordinal label-distribution learning are established.  They make ordinary smoothing especially unsafe for a rare endpoint because they explicitly move target mass inward. [Diaz & Marathe, 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html); [Wen et al., 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Ordinal_Label_Distribution_Learning_ICCV_2023_paper.html) |
| Imbalance-aware ordinal learning | SLACE explicitly targets monotonicity and balance sensitivity, and was already tested here.  Generic cost-sensitive ordinal methods and minority oversampling are also established; they do not by themselves justify a frequency-only loss. [Nachmani et al., 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34158); [George et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5217743/); [Ren et al., 2019](https://www.sciencedirect.com/science/article/pii/S0950705118306166) |
| Closest retinal/long-tail asymmetric target work | CAP-WAE uses asymmetric latent priors plus learned asymmetric soft labels for imbalanced retinal grading.  It is architecturally much broader than the present proposal, but makes an asymmetric-soft-label novelty claim untenable without a much deeper comparison. [Shaik et al., 2025](https://arxiv.org/abs/2509.26146) |

This audit did not identify an identical published objective of the form
`RPS + frequency-modulated, endpoint-neighborhood event NLL` in the sources
reviewed.  That absence is **not evidence of novelty**: a dedicated paper and
code search is still required before any manuscript claim.

## Candidate mechanisms and mechanism audit

### Candidate A — frequency-modulated predictive-mean anchoring

For `mu(p)=sum_k k p_k`, use an endpoint-only, directionally symmetric mean
correction, for example `L_RPS + lambda w_y (mu-y)^2` when `y` is an endpoint.
The frequency term `w_y` is derived from train counts; `(mu-y)^2` is the
ordinal-location component.

* **Target and direct quantity:** the strong inward predictive-mean bias; it
  directly moves `mu`, with indirect and underdetermined effects on `p_y`, CDF,
  and tail mass.
* **Why it could help / harm endpoints:** it can move class 4 upward and class
  0 downward, but it can also over-correct class 0, which is already well
  localized, if its endpoint weight is not sufficiently small.
* **Global and RPS-geometry risk:** many probability vectors share a mean.
  The correction may exchange central mass for simultaneous low/high tails or
  collapse mass into a point, neither of which preserves calibration, severe
  risk, or selective-risk ordering.  It is therefore not a distributional
  solution to a distributional problem.
* **Relationship to failed/existing methods:** it is not exactly weighted CE
  because it changes a moment rather than only `p_y`, but frequency weighting
  can reproduce its global trade-off.  It is not SLACE, yet it closely overlaps
  published mean--variance ordinal distribution learning.
* **Overconfidence / testability:** a squared mean alone need not sharpen, but
  pairing it with any variance penalty would explicitly collapse uncertainty.
  It is technically easy to test at seed 0, but scientifically weak.
* **Overlap risk:** **HIGH** — closest to the published mean component of
  mean--variance loss (Pan et al., 2018).

### Candidate B — endpoint-weighted cumulative-RPS correction

Retain CDF space and add a direction-aware reweighting of the existing RPS
threshold residuals.  With `F_j=sum_{k<=j}p_k` and `H_j(y)=1[y<=j]`, a generic
form is

\[
 L = L_{\mathrm{RPS}} + \lambda\,w_y\,\mathbf{1}\{y\in\{0,K-1\}\}
 \frac{1}{K-1}\sum_{j=0}^{K-2} a_{y,j}[F_j-H_j(y)]^2,
\]

where `a_{y,j}` is a predeclared endpoint-distance weight mirrored across the
two ends.  `w_y` is frequency-only; `a_{y,j}` is ordinal geometry.

* **Target and direct quantity:** inward CDF leakage.  For `y=K-1`, it adds
  pressure against `F_j` (mass at or below each interior threshold); for
  `y=0`, it mirrors this through `1-F_j`.
* **Why it could help / harm endpoints:** it can penalize the class-4 central
  CDF mass at every cut point, but symmetric use can pull already-good class-0
  distributions toward an excessively narrow lower tail.
* **Global and RPS-geometry risk:** its CDF representation is the most
  compatible with RPS and is less likely than a moment loss to ignore tail
  shape.  But the added weights intentionally make the score improper for the
  original conditional distribution, can distort calibration, and may trade
  accuracy/QWK for endpoint recall.
* **Relationship to failed/existing methods:** it is more ordinal than weighted
  CE and not SLACE's accumulated CE, but it is mathematically a class- and
  threshold-weighted RPS/CRPS.  It may behave like cost-sensitive weighting if
  `a` is nearly constant.
* **Overconfidence / testability:** squared CDF pressure can suppress all
  inward mass and sharpen endpoint distributions.  It is seed-0 testable, but
  requires a separate choice of the threshold profile `a`, adding avoidable
  degrees of freedom.
* **Overlap risk:** **HIGH** — weighted/balanced CRPS/RPS and cost-sensitive
  ordinal scoring are close conceptual precedents.  It is retained only as a
  fallback mechanism, not a novelty candidate.

### Candidate C — endpoint-neighborhood target-set shaping (selected)

Ordinary soft labels are rejected: at class 4 they move target mass from 4
toward 3/2 and can worsen exactly the observed shrinkage.  Instead, retain the
one-hot RPS target and add one coarse, structured target event: the outcome is
within a one-step neighborhood of its endpoint.  It is a target-distribution
shaping mechanism at the *event* level, not a smoothed replacement for the
one-hot label.

For an endpoint `y`, the event mass is
`T_y(p)=sum_{k: |k-y|<=1}p_k`.  Thus class 4 uses `p3+p4`, class 0 uses
`p0+p1`, and the definition extends to any `K>=2` without hard-coding a class.

* **Target and direct quantity:** lack of true/near-extreme mass; it directly
  increases endpoint-neighborhood tail mass, indirectly affects `p_y`, the
  CDF, predictive mean, and decisions.
* **Why it could help / harm endpoints:** for class 4, it directly rewards the
  observable missing geometry and permits meaningful `4->3/4` recovery rather
  than demanding exact class-4 mode recovery.  For class 0, it may increase
  `p0+p1` but reduce mass needed to represent genuinely ambiguous class-0
  examples or make the majority endpoint too confident.
* **Global and RPS-geometry risk:** it leaves RPS as the full-distribution
  anchor and does not force all mass to a scalar mean.  Nevertheless, it can
  hide uncertainty *within* `{3,4}` and is not a proper score for the original
  five-way conditional distribution; calibration, CDF shape, severe detection,
  and selective prediction must be treated as hard controls.
* **Relationship to failed/existing methods:** it is not weighted CE because
  its optimized quantity is a set probability rather than `p_y`; with radius
  zero it would collapse to an endpoint-weighted CE-like bonus, which is why
  radius one is fixed.  It is not SLACE, whose frequency-derived soft labels
  and accumulation affect every class.  It is adjacent to coarse-label and
  structured/soft-label supervision, but no identical RPS-plus-event objective
  was found in this focused audit.
* **Overconfidence / testability:** `-log T_y` can over-concentrate event mass,
  so it has a real overconfidence risk.  It is differentiable, uses one
  coefficient, and has a specific seed-0 prediction: upward class-4 near-tail
  mass and lower shrinkage without losing RPS risk controls.
* **Overlap risk:** **MEDIUM** — related to structured targets/label-
  distribution learning and set/coarse-label supervision, but not identified
  as an identical objective.  No novelty claim is authorized.

## Predictive mean and tail-mass conclusions

The predictive mean is a useful *diagnostic*, but insufficient as the sole
training target.  Equal means can correspond to a concentrated adjacent
distribution, a broad central distribution, or a two-tail distribution with
very different calibration, severe-error risk, and selective behavior.
Candidate A therefore has no direct control over the geometry RPS made useful.

In contrast, `T_y(p)` is a differentiable sum of Softmax probabilities.  For
endpoint radius one it is a CDF tail: `T_0=F_1` and
`T_{K-1}=1-F_{K-3}` (for `K=5`, `p0+p1` and `p3+p4`).  It is not redundant with
RPS: RPS penalizes all one-hot CDF residuals quadratically, whereas this term
penalizes failure to assign sufficient mass to one predeclared local endpoint
event logarithmically.  The latter is heuristic and intentionally changes the
population optimum, hence the strict calibration/risk controls.

## Ranking and disposition

| Rank | Candidate | Mechanistic fit | Novelty risk | Implementation complexity | Risk to RPS geometry | Seed-0 priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | C. Endpoint-neighborhood target-set shaping | HIGH | MEDIUM | LOW | MEDIUM | **PRIMARY SEED-0 CANDIDATE** |
| 2 | B. Endpoint-weighted cumulative-RPS | HIGH | HIGH | MEDIUM | LOW--MEDIUM | **BACKUP CANDIDATE** |
| 3 | A. Frequency-modulated predictive-mean anchoring | MEDIUM | HIGH | LOW | HIGH | **DEFER** |

The rank is a test-order decision, not an efficacy or novelty claim.  Candidate
C is selected because it targets the missing `p3+p4`/location geometry with one
coefficient while retaining RPS as the distributional base.  Candidate B is the
backup because its CDF geometry is attractive but its threshold-weight profile
introduces extra design choices and stronger overlap.  Candidate A is deferred
because its mean-only solution cannot distinguish the uncertainty structures
that matter here.

## Primary mathematical objective

Let `K` be the number of ordinal classes, `n_y` the training count of class
`y`, `p=softmax(z)`, and `E={0,K-1}`.  Fix the endpoint neighborhood radius
to `r=1` (no tunable radius), and define

\[
 T_y(p)=\sum_{k=0}^{K-1}\mathbf{1}\{|k-y|\le 1\}p_k,
 \qquad y\in E.
\]

Use an endpoint-relative square-root frequency modifier, normalized so the
rarest endpoint has weight one:

\[
 w_y=\sqrt{\frac{\min_{e\in E}n_e}{n_y}},\qquad y\in E.
\]

It is frequency-only.  The endpoint-neighborhood event `T_y` is the
ordinal-location component.  The per-example objective is

\[
 \ell(z,y) = \ell_{\mathrm{RPS}}(z,y)
 + \lambda\,\mathbf{1}\{y\in E\}\,w_y
 \left[-\log\bigl(\max(T_y(\operatorname{softmax}(z)),\epsilon)\bigr)\right],
\]

where `epsilon` is the dtype safety floor, not a tuned parameter.  The batch
loss is the mean of this expression.  For the current train counts, endpoint
weights are approximately `w0=0.369` and `w4=1.000`; this asymmetry is driven
solely by training support, while the geometry is reflected across endpoints.

This is intentionally **not** claimed to be proper.  RPS itself remains the
proper base loss; the extra term creates a predeclared, endpoint-local recovery
preference that must earn its place empirically.

## Hyperparameter and validation policy

The only experimental hyperparameter is `lambda`.  Predeclare the small
seed-0 diagnostic grid:

```text
lambda = {0.1, 0.3, 1.0}
```

`r=1`, square-root frequency exponent, endpoint set, and the safety floor are
fixed design choices, not searched.  The grid is not tuned on the test set.

Checkpoint selection for every lambda is minimum **validation RPS**; RPS is
available for all validation cases and protects the retained distributional
anchor.  Choose a single lambda before test evaluation using the following
predeclared validation-only rule: retain configurations whose validation RPS is
within 0.01 absolute of the seed-0 RPS baseline's selected-checkpoint validation
RPS; among them, prefer the smallest lambda with lower validation endpoint
inward shrinkage averaged macro-wise over classes 0 and 4.  If no configuration
meets the RPS guard, declare the candidate validation **NO-GO** and do not
choose by endpoint metrics.

Only six validation examples are class 4.  Consequently, validation class-4
decisions, exact recovery, and any noisy class-4-only composite must not choose
lambda or a checkpoint.  The macro endpoint shrinkage tie-break is descriptive
and must be reported with endpoint denominators; it cannot override validation
RPS.  Test results are used once, after lambda and checkpoint are fixed, for
the predeclared seed-0 GO/NO-GO decision.

## Pre-registered seed-0 evaluation and gate

Compare one finalized primary run with the frozen matched seed-0 CE and RPS
artifacts.  Report raw probabilities and mode/L1/L2 decisions, then all
existing global, decision-risk, risk-coverage, classwise, and sample-level
artifacts.  The primary decision-controlled comparison is L1, with mode and L2
as required controls.

### Class-4 geometric signal

Relative to matched seed-0 RPS, require both:

1. at least one distributional movement: `mean p4` increases by >=0.03
   absolute **or** `mean(p3+p4)` increases by >=0.05 absolute; **and**
   predictive mean rises by >=0.30 **or** inward shrinkage falls by >=0.30;
2. at least one decision/error movement: L1 `4->3/4` rises from 2/20 to at
   least 4/20 **or** L1 `4->0/1/2` falls from 18/20 to at most 16/20; class-4
   L1 MAE must not exceed the RPS value of 2.10.

This permits adjacent recovery; class-4 exact accuracy need not become positive.

### Controls for a GO TO MULTI-SEED recommendation

In addition to the class-4 signal, the candidate must stay within these
seed-0 guardrails relative to RPS (values are practical diagnostic tolerances,
not significance tests):

| Control | Required guard |
| --- | --- |
| L1 global accuracy / MAE / QWK / severe prevalence | no worse by more than 0.03 / 0.05 / 0.04 / 0.02 absolute |
| L1-risk Spearman | no decrease >0.05 |
| severe AUROC / AUPRC | no decrease >0.05 / 0.03 |
| mean ordinal-MAE selective risk | no increase >0.05 |
| Class-0 L1 accuracy / MAE / severe prevalence | no worse by more than 0.05 / 0.05 / 0.05 absolute |
| Class-0 geometry | no increase in inward shrinkage >0.10; report `p0`, `p0+p1`, and predictive mean |

The recommendation is **GO TO MULTI-SEED** only when every class-4 condition
and every guardrail holds, no numerical/probability artifact fails, and the
test evaluation was not used to select lambda or a checkpoint.  It remains a
request for explicit user approval before seeds 1--4.

### TRADE-OFF — REVIEW BEFORE MULTI-SEED

Use this disposition if class-4 geometric criteria hold but any global, class-0,
or risk guardrail fails; or if apparent class-4 movement is driven by a sharp
increase in `p3+p4` with degraded calibration/RPS/severe detection.  Do not
start further seeds automatically.

### NO-GO

Use **NO-GO** if validation fails its RPS guard; test class-4 mass/mean/
shrinkage does not materially move; only confidence changes without the
predeclared location signal; the run resembles weighted-CE global degradation;
or RPS-like risk geometry deteriorates beyond the guards.  Preserve all
negative artifacts and do not substitute decision-rule changes, temperature,
or generic focal/class weighting for this test.

## Open scientific risks

* The correction intentionally breaks strict propriety and can make endpoint
  probabilities overconfident or miscalibrated.
* It may reward adjacent mass without improving `p4` or true clinical-grade
  localization; this is why `p4`, near-tail mass, mean, shrinkage, and both
  near/far decision routes are separate endpoints.
* Endpoint-only training signal can improve rare class 4 at the expense of the
  common class 0 or middle classes; global and class-0 controls are mandatory.
* With 6 validation and 20 test class-4 examples, seed-0 estimates are noisy;
  no conclusion should be generalized without an approved multi-seed stage.
* Literature overlap remains material, particularly with structured targets,
  balanced RPS/CRPS, and recent retinal asymmetric-label work.  No novelty
  claim is supported by this audit.
