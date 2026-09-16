# Full Manuscript Consistency Audit

**Status:** Solar provenance recovery complete (2026-09-15)  
**Scope:** full main manuscript, all main tables, standalone supplement, frozen
research notes and manuscript-production audits. No experiment, reanalysis,
figure generation, claim expansion, commit, or push was performed.

## Executive verdict

**A — MANUSCRIPT CONSISTENCY AND SOLAR PROVENANCE COMPLETE.**

The manuscript is internally consistent after the corrections recorded below.
Its central claim is appropriately bounded as an empirical mechanism analysis,
not a new classifier method or a general causal account of imbalance. The
remaining items are provenance-recovery requirements for submission, rather
than contradictions in the reported claims or numbers. They do not require a
new experiment.

## Claim consistency matrix

`C` means consistent at the role appropriate to that section. `N/A` means the
topic need not be repeated there. The status column also compares the frozen
research notes and production-status record.

| Topic | Abstract | Introduction / related work | Methods | Results / discussion / limitations / conclusion | Supplement | Status | Finding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Paper positioning | C | C | C | C | C | C | Empirical mechanism analysis; not a new algorithm. |
| 2. Rare-end phenomenon | C | C | C | C | C | C | `rare-end inward localization` is the named phenomenon. |
| 3. H1 | C | C | C | C | C | C | Frozen A/C, exact-L1 rare-end MAE, negative sign favors C. |
| 4. H2a | C | C | C | C | C | C | Endpoint-specific geometry is a setting-dependent diagnostic. |
| 5. Representation/head decomposition | C | C | C | C | C | C | Historical centroid strata remain descriptive, not causal labels. |
| 6. Direction-only intervention | C | C | C | C | C | C | C changes direction while preserving representation, row norms, and biases. |
| 7. Balanced-sampling factorial | C | C | C | C | C | C | One frozen Retina RPS setting; no general causal/method claim. |
| 8. Support-severity study | C | C | C | C | C | C | Five-seed full-model grid is separate from frozen A/C and non-monotonic on localization. |
| 9. UTKFace | N/A | C | C | C | C | C | Historical supporting seed-0 evidence only; excluded from H1/H2a confirmation. |
| 10. Historical seed 0 | N/A | C | C | C | C | C | Hypothesis-forming/descriptive and not counted in any confirmatory gate. |
| 11. Confirmatory seeds 1--4 | C | C | C | C | C | C | Backbone seed is the only replication unit. |
| 12. RetinaMNIST protocol | N/A | N/A | C | C | C | C | Fixed representation; five-fold training-only OOF C evaluation; fitting-fold centroids. |
| 13. Solar protocol | N/A | N/A | C | C | C | C | Aligned training centroids and fixed archived readout; no readout-based selection. |
| 14. Geometry conclusion | C | C | C | C | C | C | Solar STRONG; Retina CE PARTIAL; Retina RPS NOT SUPPORTED. |
| 15. Novelty boundary | C | C | C | C | C | C | No claim that head direction, balancing, or geometry is novel in isolation. |
| 16. Global/UQ boundary | C | C | C | C | C | C | Localization response is not represented as global, calibration, risk, or UQ improvement. |

## Focused scientific checks

### H1

The H1 definition is identical in the protocol, main table, Results,
Discussion, Conclusion, and supplement:
\(\Delta\mathrm{MAE}=\mathrm{MAE}_C-\mathrm{MAE}_A\) for rare-end exact-L1
decisions, with a negative value favorable to C. All four CE/RPS settings have
four negative confirmatory backbone-seed effects out of four, yielding
**REPLICATED** under the frozen sign gate. Descriptive t intervals are not
substituted for that gate.

### H2a

The outcome, controls, sign orientation, LOO orientation, and verdicts agree:
Solar CE/RPS are **STRONG**, RetinaMNIST CE is **PARTIAL**, and RetinaMNIST RPS
is **NOT SUPPORTED**. H2b is consistently secondary and cannot alter H2a.
The manuscript does not treat the robust H1 response as proof of a universal
geometry mechanism.

### Historical versus confirmatory evidence and dataset roles

RetinaMNIST and Solar CE/RPS backbone seeds 1--4 are the sole confirmatory
block. Seed 0, historical B/D probes, and centroid decomposition analyses are
consistently descriptive/hypothesis-forming. UTKFace is consistently
supporting historical seed-0 evidence and is explicitly excluded from the
H1/H2a block. The UTKFace supplement table now identifies its Phase 3.13
frozen-head source so it cannot be confused with the separate Figure 1
original-CE aggregate.

### Terminology, tables, figures, and references

The manuscript uses the controlled vocabulary appropriately:
`rare-end inward localization` for the phenomenon, `direction-only adaptation`
for C, `classifier-head direction` for the actionable component,
`endpoint-specific representation geometry` for H2a, and `setting-dependent
diagnostic` for its interpretation. Natural descriptive variants such as
“rare upper endpoint” identify the class role rather than a competing claim.

All four figures are included with canonical-file captions, and their labels
are resolved by compilation. The obsolete in-source figure-insertion TODO
comments were removed; they were implementation comments, not missing evidence
or citation placeholders. Main table verdict spelling is now matched to the
supplemental matrix. No undefined `\ref`, citation, or `\cite{TODO}` token was
found after compilation.

## Corrections made

Three issue categories, comprising nine small textual corrections, were made:

1. Removed four stale Figure 1--4 insertion TODO comments now that canonical
   figures are actually included.
2. Replaced the claim that all Solar configuration/checkpoint material is
   supplied with an availability-qualified statement and an explicit pointer to
   the supplement's two Solar gaps.
3. Standardized frozen verdict labels in the main/supplement tables and named
   the distinct Phase 3.13 source for the UTKFace supplemental table.

No aggregate number, protocol, figure, scientific decision, or claim boundary
was changed.

## Provenance gap triage

| Gap | Classification | Current manuscript handling | Submission action |
| --- | --- | --- | --- |
| Solar per-seed backbone configuration tree/manifests are absent locally. | **CLOSED** | Exact archived paths, selected epochs, validation losses, READY status, and protocol-level configuration are indexed in the Solar confirmatory provenance manifest. | No further recovery required for the provenance gate. |
| Solar seed-level H2a coefficients, HC3 intervals, and LOO deltas are retained locally. | **CLOSED** | Checksummed CE/RPS per-seed tables and summaries are indexed in the Solar H2a provenance artifact. | No further recovery required for the provenance gate. |
| Solar aligned train/validation population counts are absent from a local frozen manifest. | **CLOSED** | The retained Phase 3.8 alignment audit verifies 45,047/2,431/28,006. | No further recovery required. |

The prior traceability gaps are closed; their recovery does not change the
existing H1 or frozen H2a numerical conclusions.

## Remaining venue-readiness items

1. Recover/archive the saved Solar seed-level H2a artifacts without rerunning analysis.
2. Adopt the target venue template, then perform page-budget, anonymity, and
   template-specific bibliography/figure-layout review.
3. Retain the current claim hierarchy during formatting; do not turn the
   direction probe into a proposed method claim.

**Final audit verdict: C — MANUSCRIPT CONSISTENCY COMPLETE; SOLAR H2a
PROVENANCE GAP REMAINS.**
