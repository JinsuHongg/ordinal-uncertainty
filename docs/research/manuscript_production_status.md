# Manuscript Production Status

## Current status

**ICLR 2027 submission formatting pass complete; main text is 4 pages and
anonymous, with integrated appendix after references.** See
`docs/research/iclr2027_submission_audit.md`.

**Final readiness pass:** restored core protocol and H2a explanation; main text
is now 6 pages, references begin on page 6, and the appendix is integrated
after references. Content and anonymity are ready pending author confirmation
of the AI-use disclosure and a decision on anonymous code/artifact release; see
`docs/research/iclr2027_final_readiness_review.md`.

**Introduction/Related Work restoration:** the main text is now 7 pages and
references begin on page 7. The restored motivation and compact positioning use
only verified existing citations; see
`docs/research/iclr2027_intro_relatedwork_restoration.md`.

**Confirmatory global trade-off analysis:** all 16 saved A/C conditions show
endpoint improvement with average global-MAE cost and macro-MAE improvement.
The ICLR Results now records this redistribution boundary; see
`docs/research/confirmatory_global_tradeoff_analysis.md`.

**Retina confirmatory feature regeneration:** deterministic frozen 512-D
feature archives for CE/RPS seeds 1–4 are now available and reproduce archived
A outputs exactly. This makes the separately authorized Retina natural-sampling
head follow-up technically possible; it does not change manuscript claims.
Solar remains separate cluster work. See
`docs/research/retina_confirmatory_frozen_feature_regeneration.md`.

**Retina natural-sampling follow-up:** natural direction-only N improves the
endpoint in only 2/4 CE and 2/4 RPS seeds, while balanced C improves every
archived seed. The Retina wording is consequently narrowed to a balanced
direction-only response. A bias/prior-only control remains a lower-priority
reviewer concern; Solar remains separate. See
`docs/research/retina_natural_sampling_direction_only_analysis.md`.

**Solar RPS natural-sampling follow-up:** natural N improves X exact-L1 MAE in
3/4 validated frozen-RPS seeds but remains weaker than balanced C in all four
and does not reproduce C's broad endpoint-mass redistribution. This is a
Solar-RPS-only mixed descriptive result; Solar CE is unresolved and untouched.
It does not change frozen H1/H2a or manuscript wording absent separate writing
authorization. See `docs/research/solar_rps_natural_sampling_direction_only_analysis.md`.

**Solar CE feature-identity audit:** direct scheduler and runtime records now
verify historical CE A/C execution on V100 and recent CE regeneration on A30.
The 64-row strict-FP32 A30 audit supports a TF32 contribution, but the later
full strict-FP32 identity gate still exceeded frozen numerical tolerances;
therefore V100-versus-A30 is not yet a complete exclusive causal explanation.
CE N fitting remains blocked. See
`docs/research/solar_ce_gpu_provenance_audit.md` and
`docs/research/solar_ce_feature_identity_mismatch_audit.md`.

**Solar CE strict-FP32 full regeneration:** all four CE jobs preserved exact
IDs, labels, mode, and L1 decisions, but full-readout logit/probability errors
still exceeded frozen identity tolerances. No CE feature archive was promoted
and CE N fitting remains blocked. This is provenance/debug status only and
does not change manuscript scientific claims. See
`docs/research/solar_ce_strict_fp32_feature_regeneration.md`.

**Full consistency audit complete; Solar provenance recovery complete
(2026-09-15).** The manuscript uses an article-compatible anonymous layout so
that a conference template can be adopted later without changing the frozen
claim structure. The audit verdict is **A — MANUSCRIPT CONSISTENCY AND SOLAR
PROVENANCE COMPLETE**; see
`docs/research/manuscript_full_consistency_audit.md`.

## Completed in this phase

- Created `manuscript/` with the main source, section layout, table directory,
  figure/supplement directories, and an empty verified-reference placeholder.
- Drafted `manuscript/sections/results.tex` using the frozen cross-setting
  claim hierarchy.
- Created frozen H1 and H2a main-text summary tables.
- Mapped canonical Figures 1--4 to the Results section without copying or
  regenerating figure artifacts.
- Completed the first Discussion draft, separating the robust H1 direction-only
  response from the setting-dependent H2a geometry diagnostic.
- Completed first drafts of the Introduction and Related Work, using three
  claim-locked contributions and the mechanism/empirical-analysis positioning.
- Completed first drafts of the Abstract, Conclusion, and Limitations.
- Completed first drafts of Problem Setup and Experimental Protocol, including
  the frozen A/C intervention, data/evidence roles, OOF and centroid
  provenance, and H1/H2a replication rules.
- Added a Methods/Experimental Setup completeness audit and explicit
  main-text-versus-supplement training-detail classification.
- Completed citation verification and replaced all six manuscript citation
  placeholders with nine verified bibliography entries. The Introduction was
  minimally split to align long-tail intervention and geometry claims with
  their direct sources.
- Completed the first standalone supplement draft with configuration and Solar
  provenance tables, confirmatory seed-level H1/H2a material, historical
  mechanism context, UTKFace supporting evidence, and secondary diagnostics.
- Completed the title audit and selected the working title
  *Classifier-Head Contributions to Rare-End Inward Localization in Imbalanced
  Ordinal Classification*.
- Completed the full manuscript consistency audit across the main text,
  tables, supplement, and frozen research notes. It resolved stale figure TODO
  comments, standardized frozen verdict labels, clarified the Phase 3.13
  UTKFace supplemental source, and made the Solar provenance caveat explicit
  in the main protocol.
- Recovered Solar confirmatory checkpoint/configuration provenance for all
  eight CE/RPS seed-1--4 conditions from existing integrity and A/C manifests,
  and recovered explicit aligned counts (45,047/2,431/28,006) from the retained
  alignment audit. The provenance index is recorded under the Solar mechanism-
  replication output tree.
- Verified the locally copied Solar saved H2a analysis tree for all eight
  CE/RPS seed-level confirmatory results. It contains the per-seed
  coefficients, HC3 intervals, LOO MSE pairs, and positive LOO deltas; a
  checksummed provenance index now records the source files without
  recomputation.

## Evidence-tier controls in the draft

- RetinaMNIST and Solar CE/RPS seeds 1--4 are the primary confirmatory H1/H2a
  evidence block.
- Historical seed-0 A/C results are labeled hypothesis-forming.
- UTKFace is mentioned only as historical supporting evidence in the Figure 1
  caveat and is excluded from the confirmatory tables.
- H2a is written as a setting-dependent diagnostic; H2b is not used to alter
  its conclusion.

## Open provenance items before submission

- No material Solar provenance gap remains. Preserve the checksummed H2a
  source tables and do not recompute or tune them.

## Next writing phase

Adopt the target venue template and perform the page budget, anonymity,
bibliography, and figure-layout pass only under separate authorization. Do not
add experiments or change the frozen H1/H2a claim boundaries during formatting.
