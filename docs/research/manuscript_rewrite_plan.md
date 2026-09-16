# Manuscript Rewrite Plan and Claim-Lock Audit

## 1. Executive decision

**Position the submission as a mechanism / empirical analysis paper.**  The
paper is ready for manuscript writing without another experiment, provided its
central claim is the replicated direction-only localization response and not a
new classifier method or universal geometry explanation.

No manuscript source (`.tex`, `.qmd`, `.md`, or `.rst` paper draft) is present
in the repository.  This audit therefore treats
`docs/research/paper_story_architecture.md` and the four current figure captions
as the claim-bearing manuscript planning sources.  A requested
`docs/research/solar_h1_h2_confirmatory_analysis.md` file is not present; the
frozen Solar verdicts are taken from the completed cross-setting synthesis and
saved confirmatory artifacts.

## 2. Final paper thesis

The supplied candidate is nearly correct, but “show a reproducible
inward-localization response” can be read as adaptation causing the phenomenon.
Use this tighter thesis instead:

> **Rare upper extremes exhibit inward localization in the studied imbalanced
> ordinal settings.  Across RetinaMNIST and Solar under CE- and RPS-trained
> representations, a fixed-norm, fixed-bias direction-only classifier-head
> intervention reproducibly reduces rare-end localization error, whereas
> endpoint-specific representation geometry has setting-dependent explanatory
> value.**

The first sentence names the phenomenon; the second names the intervention
response and its boundary.  It does not imply that the intervention is a new
algorithm, that imbalance is the sole cause, or that geometry universally
explains recovery.

## 3. Contribution hierarchy

Use **three** Introduction contribution bullets.  A fourth bullet for the
factor/severity studies would over-weight bounded Retina-only evidence.

1. **Phenomenon:** characterize rare upper-end inward localization using
   routing, ordinal distance, predictive location, and shrinkage in the
   studied ordinal settings.
2. **Replicated head response:** separate frozen representation and head
   diagnostics, then show that the fixed-norm, fixed-bias direction-only probe
   improves rare-end localization in every confirmatory seed across Retina CE,
   Retina RPS, Solar CE, and Solar RPS.
3. **Boundaries:** show that endpoint-specific geometry is a
   setting-dependent diagnostic, while the controlled Retina studies identify
   balanced sampling as the dominant observed training signal and reject a
   clean support-severity dose-response law.

The historical mixed representation/head decomposition belongs inside
contributions 1–2 as a diagnostic motivation, not as independently replicated
causal proof.  The Retina factorial and severity studies belong under
contribution 3, not as new methods.

## 4. Section-by-section audit

| Planned/source section | Status | Action and reason |
| --- | --- | --- |
| Title | **MAJOR REVISION** | Replace any UQ- or method-first framing with rare-end localization and classifier-head mechanism framing. |
| Abstract | **ADD** | No manuscript abstract source exists; use the claim-locked blueprint below. |
| Introduction | **MAJOR REVISION** | Lead with ordinal endpoint localization, not generic UQ or a proposed correction. Use three bounded contributions. |
| Related Work | **MAJOR REVISION** | Explicitly distinguish from cRT/balanced retraining, angular/norm methods, feature/classifier alignment, and ordinal imbalance methods; do not claim those components are novel. |
| Problem setup and diagnostics | **MINOR REVISION** | Define predictive location, inward shrinkage, rare-end routing, and A/C as diagnostic protocol rather than a Method. |
| Historical phenomenon/decomposition | **MOVE** | Present the historical seed-0 evidence before confirmatory results, clearly labeled hypothesis-forming/descriptive. |
| Confirmatory results | **ADD** | Add a dedicated multi-seed Retina/Solar H1/H2 section and a compact table; this is now the primary evidence block. |
| Factorial and severity studies | **MOVE** | Place after H1/H2 as boundary studies, not as support for a causal chain or method selection. |
| Discussion | **MAJOR REVISION** | Interpret robust H1 separately from setting-dependent H2a; foreground limits and prior-art boundary. |
| Limitations | **ADD** | Make dataset count, endpoint support, Solar readout, UTKFace status, centroid coarseness, and lack of causal imbalance identification explicit. |
| Conclusion | **MINOR REVISION** | State the bounded mechanism takeaway only. |
| Figures/captions | **RECAPTION / REORDER** | Keep the existing visual assets pending authorization, but revise their role/order around confirmatory H1 and H2a. |
| Supplement | **ADD** | Move historical UTKFace, full controls, global/UQ metrics, exhaustive per-seed outputs, and extended geometry diagnostics here. |

### Claim-consistency flags in current planning artifacts

No manuscript prose exists to quote.  The following exact claim-bearing text
should be revised when prose/captions are assembled; these are correction
directions, not edits in this task.

| Source and current wording | Risk | Required correction direction |
| --- | --- | --- |
| `paper_story_architecture.md`: “Across three studied ... settings ... this failure is mixed” | C2 is historical/supporting only, and UTKFace is outside the main confirmatory block. | State that the mixed decomposition is historical descriptive evidence; reserve the primary replicated claim for H1 in Retina/Solar. |
| `paper_story_architecture.md`: C3 includes “plus UTKFace RPS.” | Can imply equivalent confirmation. | Move UTKFace to supplement and label seed-0 historical/supporting evidence. |
| `paper_story_architecture.md`: “direction is the most consistent actionable head axis across these settings” | Safe only if “tested controls” and the main evidence block are specified. | Use “most consistently reproduced head-level response among the tested controls in RetinaMNIST and Solar.” |
| Figure 2 caption: “failure contains both representation- and head-level components.” | Reads as a general mechanism law. | Use “historical diagnostics identify representation-inward and head-mislocalized components” and retain the descriptive/centroid caveat. |
| Figure 3 caption: “Across all four within-setting A→C comparisons ...” | The visual uses historical seed-0 summaries; it does not itself display the new seed 1–4 confirmation. | Keep the visual as an illustrative mechanism plot; point to a new per-setting confirmatory table for the 4/4 H1 result. |
| Figure 1 caption: “across the studied ordinal tasks.” | Permissible only if the UTKFace role is visibly qualified. | Note in caption or nearby text that UTKFace is historical supporting evidence; do not call it confirmatory. |
| Any future “geometry explains recovery across datasets” wording | Contradicts H2a. | Report Solar strong, Retina CE partial, Retina RPS unsupported; call geometry setting-dependent. |
| Any future “our method,” “new classifier,” or “imbalance causes” wording | Violates novelty and causal boundaries. | Call A/C a controlled diagnostic intervention/probe and describe studied imbalanced settings. |

## 5. Results redesign

Merge historical direction evidence with the confirmatory story only through
clear role labels.  The confirmatory H1/H2 result should appear before the
factorial/severity boundaries and should not be buried after seed-0 evidence.

| Subsection | Main question | Evidence / primary artifact | Strongest allowed conclusion | Must not conclude |
| --- | --- | --- | --- | --- |
| **4.1 Rare upper-end inward localization** | What ordinal failure is observed? | Figure 1; CE seed-0 phenomenon summaries for Retina/Solar; historical supporting UTKFace | The studied settings show upper-end predictions and predictive locations shifted inward, with dataset-dependent severity. | Imbalance universally causes it or raw severity is comparable across datasets. |
| **4.2 Historical representation/head decomposition** | Is all inward localization equally head-actionable? | Figure 2, historical CE centroid groups | Historical diagnostics distinguish representation-inward from rare-end-like but head-mislocalized cases. | Centroid routing causally proves collapse or irrecoverability. |
| **4.3 Controlled direction-only probe** | What head component is most consistently actionable in the original controls? | A/B/C/D seed-0 summaries; Figures 2–3 | With backbone, norms, and biases fixed, changing direction can move rare-end localization outward. | C is a deployable universally better classifier. |
| **4.4 Confirmatory H1 across objectives and datasets** | Does the A→C direction response persist across new backbone realizations? | **New main Table 2** from frozen H1 summaries; Figure 3 as illustrative seed-0 context | All four new seeds improve rare-end MAE in each Retina CE/RPS and Solar CE/RPS setting. | “16 independent experiments,” objective equivalence, or external/universal validation. |
| **4.5 Endpoint-specific geometry as a bounded diagnostic** | Does geometry add to original-head state and generic geometry? | **Table 2, H2a block**; optional supplemental per-seed coefficients | Strong in Solar; partial in Retina CE; unsupported in Retina RPS. | Geometry is a universal mechanism, or H2b rescues H2a. |
| **4.6 Controlled Retina factor study** | Which observed training signal accompanies the beneficial direction change? | Figure 4A–B; Phase 3.19 table | In the one-backbone frozen-RPS factorial, balanced sampling is the dominant observed factor. | Balanced sampling is a new method or cross-dataset causal law. |
| **4.7 Rare-support severity boundary** | Does lower rare support yield a monotonic localization law? | Figure 4C–E; Phase 3.20A summary | Direct p4/margin evidence weakens, but localization severity is non-monotonic and seed-sensitive. | Support alone determines localization severity. |

**Recommended new main Table 2 — Confirmatory direction and geometry summary**

| Dataset | Objective | H1 negative ΔMAE seeds / 4 | Mean ΔMAE (SD) | H1 verdict | H2a beneficial beta seeds / 4 | H2a positive ΔLOO seeds / 4 | H2a verdict | Evaluation population |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| RetinaMNIST | CE | 4 | -0.6553 (0.5130) | Replicated | 4 | 2 | Partial | training-only OOF |
| RetinaMNIST | RPS | 4 | -0.4280 (0.2326) | Replicated | 2 | 3 | Not supported | training-only OOF |
| Solar | CE | 4 | -0.923 (0.268) | Replicated | 4 | 4 | Strong | fixed archived readout |
| Solar | RPS | 4 | -1.031 (0.164) | Replicated | 4 | 4 | Strong | fixed archived readout |

Table 1 should remain a dataset/protocol/support table.  Put per-seed H2a
coefficients, HC3 intervals, ΔLOO values, all endpoint routing, and collateral
metrics in the supplement.

## 6. Discussion redesign

### 6.1 What is robust

Opening sentence: **“The robust result is a repeated head-level localization
response, not a universal classifier correction.”**

- H1 has the beneficial sign in all four frozen confirmatory seeds within each
  Retina/Solar CE/RPS setting.
- The effect concerns rare-end localization under the exact A/C protocol.
- Direction-only preserves frozen features, original row norms, and biases.
- Localization gains do not establish global, calibration, UQ, or deployment
  improvement.

### 6.2 What is not universal

Opening sentence: **“The endpoint-geometry diagnostic is informative in some
settings but does not define a general law of recovery.”**

- Solar H2a is strong under both objectives.
- Retina CE is partial and Retina RPS fails the frozen H2a gate.
- Report the negative result without post-hoc metric selection.
- Do not infer causal representation limitation from non-recovery.

### 6.3 What the intervention establishes

Opening sentence: **“The intervention isolates a head-actionable component;
it does not identify a universally deployable correction.”**

- A/C holds the backbone, original norms, and biases fixed.
- It shows that some localization error can change at the head level.
- It does not establish that direction is the sole cause of initial error.
- It does not repair all representation-inward cases or guarantee collateral
  safety.

### 6.4 Relation to long-tail literature

Opening sentence: **“The contribution is the ordinal localization object and
controlled diagnostic, rather than retraining or classifier geometry itself.”**

- Acknowledge cRT/balanced head retraining, norm/angle effects, and
  feature/classifier alignment as established.
- Distinguish ordinal endpoint routing and predictive-location displacement
  from nominal tail accuracy.
- Describe direction-only adaptation as a probe, not a proposed algorithm.
- Make the literature boundary explicit in Related Work and Discussion.

### 6.5 Why H2a setting dependence matters

Opening sentence: **“Retaining the preregistered H2a boundary makes the
mechanistic interpretation more credible, not less.”**

- The same criterion was applied to all four settings.
- Strong Solar and mixed Retina outcomes prevent universal overclaiming.
- H2b remains secondary and cannot alter the primary conclusion.
- The result constrains future theory rather than inviting metric fishing.

### 6.6 Limitations

Opening sentence: **“The evidence is deliberately narrow and should be read as
a reproducible study of two confirmatory settings rather than a universal
account of ordinal imbalance.”**

- Only RetinaMNIST and Solar belong to the multi-seed confirmatory block.
- UTKFace has seed-0 historical/supporting evidence only.
- Centroid geometry is a coarse diagnostic; endpoint support differs sharply.
- Solar uses a fixed archived readout; it is not a new external dataset per
  seed.
- The support grid does not causally identify class imbalance or a monotonic
  severity law.

## 7. Introduction redesign

Use five paragraphs:

| Paragraph | Role and content | Transition |
| --- | --- | --- |
| 1 | Establish that ordinal endpoints make error direction meaningful: an upper endpoint routed inward differs from generic tail inaccuracy. | Motivate why aggregate accuracy/MAE can obscure this failure. |
| 2 | Introduce rare upper-end inward localization through routing, predictive mean, and shrinkage; describe it as observed in studied imbalanced settings, not caused by imbalance. | Ask whether the failure is representational, head-level, or mixed. |
| 3 | Set up frozen representation and classifier-head decomposition; factor a head into direction, norm, and bias. Define direction-only adaptation as a controlled diagnostic. | Clarify that this is not a new classifier method. |
| 4 | State the evidence arc: historical decomposition, replicated H1 in Retina/Solar CE/RPS, and setting-dependent H2a. | Separate robust effect from explanatory diagnostic. |
| 5 | Give the three contribution bullets and boundaries: factor/severity studies constrain interpretation; no universal causal imbalance or correction claim. | Lead into related work and diagnostic protocol. |

The Introduction should de-emphasize uncertainty quantification: retain
uncertainty/risk as context and a boundary, not as the central contribution.
Do not call the diagnostic protocol a “method.”

## 8. Abstract blueprint

1. **Problem:** Rare ordinal endpoints can be especially consequential, yet
   aggregate performance can obscure whether their errors are displaced inward.
2. **Gap:** Existing long-tail and ordinal analyses do not by themselves
   isolate the head-actionable component of rare-end localization.
3. **Analysis:** We study rare upper-end inward localization using frozen
   representation/head diagnostics and fixed-norm, fixed-bias direction-only
   adaptation.
4. **H1:** Across RetinaMNIST and Solar under CE and RPS, all four
   confirmatory backbone seeds per setting reduce rare-end L1 localization
   error under the direction-only intervention.
5. **H2a:** Endpoint-specific geometry adds explanatory value strongly in
   Solar, partially in Retina CE, and not under the preregistered criterion in
   Retina RPS.
6. **Boundary:** A controlled Retina factorial identifies balanced sampling as
   the dominant observed signal, while support reduction yields no clean
   monotonic localization dose response.
7. **Takeaway:** Rare-end localization has a reproducible head-actionable
   component in the studied settings, but its geometry-based explanation is
   setting-dependent.

## 9. Title audit

| Candidate | Strength | Risk | Recommend |
| --- | --- | --- | --- |
| **Rare-End Inward Localization and Classifier-Head Mechanisms in Imbalanced Ordinal Classification** | Names phenomenon, setting, and analysis without method claim. | Long but precise. | **Yes — preferred** |
| Rare-Extreme Localization Bias in Imbalanced Ordinal Classification | Clear phenomenon-first framing. | Omits head mechanism. | Yes, concise fallback |
| Understanding Rare-End Localization Failure in Ordinal Classification | Readable and conservative. | Omits imbalance and mechanism specificity. | Yes, fallback |
| Classifier-Head Mechanisms for Rare-End Localization under Ordinal Imbalance | Leads with mechanism. | Can imply a correction method. | Maybe |
| Beyond Tail Accuracy: Rare-End Localization in Imbalanced Ordinal Classification | Strong contrast with long-tail accuracy literature. | “Beyond” can sound rhetorical; omits mechanism. | No, secondary option |

## 10. Figure and table mapping

| Artifact | Disposition | Results role | Required planning change |
| --- | --- | --- | --- |
| Figure 1 | **RECAPTION** | 4.1 phenomenon | Preserve the CE baseline visual; identify UTKFace as historical supporting evidence in nearby text/caption. |
| Figure 2 | **RECAPTION** | 4.2 historical decomposition | Use “historical diagnostic” / “audited centroid grouping”; do not headline universal mixed failure. |
| Figure 3 | **RECAPTION + REORDER** | 4.3 illustrative A→C mechanism context | Place adjacent to 4.4 but do not let seed-0 visual stand in for confirmatory H1. Retain population-role caveat. |
| Figure 4 | **KEEP AS IS** | 4.6–4.7 boundaries | Retain separate-regime labeling and non-monotonic traces; keep after confirmatory H1/H2. |
| Table 1 (new/assembled) | **ADD** | Setup | Dataset, ordered classes, split role, rare endpoint support, backbone/objective, and confirmatory role. |
| Table 2 (new/assembled) | **ADD** | 4.4–4.5 | Four-setting H1/H2a summary defined above. |
| Full metrics/per-seed outputs | **MOVE TO SUPPLEMENT** | Reproducibility | Include H2a coefficients/intervals, collateral metrics, A/B/C/D details, and full severity grid. |

The current four figures are sufficient for the planned narrative only after
recaption/reordering and addition of the new confirmatory summary table.  They
are not sufficient by themselves because no current figure reports the four
new seeds and frozen H2a verdicts compactly.

## 11. UTKFace placement

Place UTKFace in a **supplementary “Historical Supporting Evidence”** section,
after the primary Retina/Solar confirmatory analysis.  Mention it once in the
main text’s data/provenance paragraph as a historical observation, not in the
main H1/H2 table or Abstract evidence count.

- Exact label: **Historical supporting evidence (UTKFace, seed 0)**.
- Exact caveat: **“UTKFace has provenance-clean CE/RPS seed-0 artifacts but no
  seed-1–4 checkpoint set; it is not part of the multi-seed confirmatory
  RetinaMNIST/Solar evidence block.”**

## 12. Terminology freeze

| Preferred term | Avoid | Reason |
| --- | --- | --- |
| rare upper extreme / rare upper endpoint | tail class, extreme class interchangeably | Keeps ordinal endpoint meaning explicit. |
| rare-end localization | tail accuracy | Names ordinal location, not generic classification accuracy. |
| inward localization | collapse (unqualified) | “Collapse” implies a stronger causal representation statement. |
| direction-only adaptation | our method / proposed algorithm | It is a controlled diagnostic intervention. |
| classifier-head direction | classifier weight, when scale is separated | Preserves the norm/direction distinction. |
| endpoint-specific representation geometry | geometry explains recovery | H2a is setting-dependent. |
| setting-dependent diagnostic | universal recovery mechanism | Matches frozen H2a verdicts. |
| confirmatory backbone seed | independent experiment/dataset | Seeds are backbone realizations within a setting. |
| historical / hypothesis-forming / supporting | confirmatory, for UTKFace seed 0 | Preserves evidence tier. |
| fixed archived Solar readout | independent external test per seed | Captures its fixed population and reuse role. |
| studied imbalanced ordinal settings | imbalance causes | Avoids causal identification claim. |

## 13. Claim-level wording

1. **Abstract main result:** “Across RetinaMNIST and Solar under CE- and
   RPS-trained representations, direction-only adaptation reduced rare-end L1
   localization error in all four confirmatory backbone seeds per setting.”
2. **Introduction contribution:** “We use a fixed-norm, fixed-bias
   direction-only head intervention as a controlled probe of the
   head-actionable component of rare-end localization.”
3. **Results H1 conclusion:** “The A→C localization response replicated in all
   four frozen seeds in each tested dataset-objective setting.”
4. **Results H2 conclusion:** “Endpoint-specific geometry was strongly
   informative in Solar but only partial in Retina CE and unsupported in
   Retina RPS under the preregistered H2a criterion.”
5. **Discussion interpretation:** “The intervention demonstrates a reproducible
   head-actionable component under the tested protocol; it does not establish a
   universally effective classifier correction or a causal account of all
   representation failure.”
6. **Limitation:** “Our multi-seed confirmatory evidence covers two datasets;
   UTKFace is provenance-clean historical supporting evidence rather than an
   equivalent replication.”
7. **Conclusion:** “Rare-end inward localization has a reproducible
   direction-responsive head component in the studied settings, while its
   geometry-based explanation is setting-dependent.”

## 14. Novelty-risk corrections

| Risky framing | Why risky | Corrected framing |
| --- | --- | --- |
| Balanced classifier retraining is the contribution | cRT/balanced head retraining is established. | Balanced sampling is a bounded observed training signal in one factorial. |
| Decoupled head retraining is novel | Established long-tail practice. | The controlled probe isolates rare-end ordinal localization while holding features/norms/biases fixed. |
| Classifier direction/norm is novel | Angular/norm effects are established. | Direction is the most consistently reproduced axis among tested controls. |
| Feature/classifier misalignment is new | Known long-tail theme. | The paper studies its ordinal endpoint-localization manifestation. |
| Long-tail geometry distortion explains recovery | H2a is not general. | Endpoint-specific geometry is a setting-dependent diagnostic. |

## 15. ICLR-oriented positioning

Choose **1. Mechanistic analysis paper**, with empirical confirmation as its
evidence base.  Do not position it as a method paper or a method/phenomenon
hybrid whose intervention is expected to beat baselines globally.

- **Foreground:** rare-end inward localization, the frozen A/C diagnostic, the
  four-setting multi-seed H1 result, and the explicit H2a boundary.
- **Secondary:** historical mixed decomposition, factor/severity studies,
  scale/bias controls, and risk/UQ outcomes.
- **Supplement:** UTKFace seed-0 evidence, complete per-seed tables,
  checkpoint/provenance details, all collateral metrics, extended geometry
  diagnostics, and stopped branches.

## 16. Rewrite order / execution checklist

1. **Results:** establishes the evidence hierarchy and prevents later prose
   from overselling H2a.
2. **Discussion:** fixes interpretation, prior-art boundary, and limitations
   from the Results facts.
3. **Introduction:** state only the contributions that Results supports.
4. **Related Work:** make overlap exclusions explicit after the contribution
   hierarchy is fixed.
5. **Abstract:** compress the now-locked thesis and H1/H2 boundary.
6. **Title:** select wording that matches the final abstract rather than a
   prior method/UQ direction.
7. **Conclusion:** mirror the bounded thesis.
8. **Limitations:** make evidence tier and population constraints explicit.
9. **Captions:** align Figure 1–4 wording with historical versus confirmatory
   status and point Figure 3 to Table 2.
10. **Supplement organization:** move full diagnostics and UTKFace supporting
    evidence after all main-text claim roles are stable.

This order keeps the manuscript evidence-led: the abstract, title, and
contribution bullets are written only after the results hierarchy and its
limits are fixed.
