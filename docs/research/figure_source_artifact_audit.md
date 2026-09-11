# Paper Figure Source-Artifact Audit

## Scope and verdict

**Audit date:** 2026-09-10. This is a source-artifact, provenance, and
availability audit plus deterministic data-consolidation record. No model was
run, no artifact was regenerated, and no figure was created.

**Verdict: A — FIGURE SOURCES READY FOR DATA CONSOLIDATION.** All proposed
main-figure panels have local, machine-readable sources. Most panels require a
deterministic extraction (for example, parsing stored probability vectors,
recomputing exact discrete L1 decisions, or joining by sample ID); this is not
new evaluation. The Figure 2 strata must be recomputed from their saved
train-feature centroids and joined by ID before any sample-level table is
emitted.

## Canonical table outputs and verification

Canonical tables were generated only from the paths and transformations
specified below by `scripts/build_paper_figure_data.py`. They are intentionally
ignored experiment/manuscript outputs, stored at
`outputs/manuscript/figure_data/`:

- `figure1_endpoint_samples.csv`, `figure1_endpoint_summary.csv`
- `figure2_mechanism_samples.csv`, `figure2_mechanism_summary.csv`
- `figure3_direction_summary.csv`
- `figure4_factorial_summary.csv`, `figure4_severity_per_seed.csv`,
  `figure4_severity_summary.csv`
- `verification.json`

The generation run completed with **PASS**: all 55 aggregate, ID-alignment,
and Phase 3.20A aggregation checks passed at absolute tolerance `1e-9`.
`verification.json` records each comparison. This does not authorize plotting,
new experiments, or an expansion of the manuscript claims.

Status definitions:

- **READY:** the saved table/JSON directly contains the planned aggregate.
- **READY_WITH_TRANSFORMATION:** the source contains all inputs but needs a
  documented deterministic derivation or ID-keyed join.
- **PARTIALLY_SUPPORTED:** a proposed aggregate exists but lacks one required
  source field or alignment proof.
- **BLOCKED:** no safe machine-readable source exists. No proposed main panel
  is blocked in this checkout.

## Audit protocol and non-negotiable provenance checks

Every future extraction must record dataset, objective/representation,
condition, split role, seed(s), checkpoint lineage, population size, rare-end
support, and the exact discrete L1 Bayes decision rule. It must build
`saved raw artifact -> figure-specific table -> plot`; manuscript numbers must
not be hand-copied from phase notes when a raw artifact exists.

For all Figure 2 joins, first assert: unique IDs in each source; identical
rare-population ID sets; identical labels; expected objective and split; and
the stated centroid definition. Join only on `sample_id`; positional alignment
is prohibited. Figure 2 Retina uses the Phase 3.18A **training-only OOF**
population (`n=1,080`, C4=`66`), never the Phase 3.3 test C4=`20` population.
Solar uses the retained aligned future-period test population (`n=28,006`,
X=`921`).

The audit directly checked that Retina 3.18A A/C prediction CSVs each contain
1,080 unique IDs, have identical ID sets, and have equal labels. It directly
checked that Phase 3.20A's aggregate CSV contains all 25 unique
`(seed, n4)` cells: supports `66,50,33,16,8` crossed with seeds `0--4`.
The saved completeness manifest reports the same 25 cells. The compressed NPY
members in the feature/prediction archives expose IDs, labels, logits, and
probabilities; their cross-archive equality still must be asserted during the
deterministic extraction run.

## Source matrix

| Panel | Status | Scientific message | Primary artifact(s) | Population / caveat |
| --- | --- | --- | --- | --- |
| 1A | READY_WITH_TRANSFORMATION | Original CE rare endpoint routes inward | Retina resolution check, UTK CE predictions, Solar CE Phase 3.8 NPZ | Original seed-0 CE; exact L1 recomputed for Retina. |
| 1B | READY_WITH_TRANSFORMATION | CE probability location is inward | same prediction sources | Derive predictive mean and shrinkage from probabilities. |
| 1C | READY_WITH_TRANSFORMATION | Upper endpoint is harder than lower endpoint | same sources plus Solar metrics | Recompute L1 MAE from per-sample IDs/labels/decisions. |
| 2A | READY_WITH_TRANSFORMATION | Rare examples separate into centroid strata | Retina 3.3 CE features; Solar 3.9 CE train/test features | Recompute raw-Euclidean train centroids and join IDs. |
| 2B | READY_WITH_TRANSFORMATION | Original CE head maps both strata inward | 3.18A A CSV; Solar 3.18B A NPZ | ID-keyed join to 2A strata. |
| 2C | READY_WITH_TRANSFORMATION | CE direction recovery is concentrated in rare-end-like stratum | 3.18A A/C CSVs; Solar 3.18B A/C NPZ | Exact fraction/count derived after join; descriptive only. |
| 2D | READY_WITH_TRANSFORMATION | A→C probability location movement differs by stratum | same as 2B/2C | Derive `mu`, `S`, and per-sample deltas. |
| 3A | READY_WITH_TRANSFORMATION | A→C lowers rare-end MAE in four settings | Retina 3.10C/3.18A; Solar 3.15/3.18B | Retina OOF; Solar archived confirmatory readout. |
| 3B | READY_WITH_TRANSFORMATION | A→C exact rare-end fraction changes | same | Plot fraction; annotate raw count only. |
| 3C | READY_WITH_TRANSFORMATION | A→C reduces shrinkage | same | Derive/verify from probabilities. |
| 3D | READY_WITH_TRANSFORMATION | A→C increases rare-vs-adjacent logit margin | same plus saved margin CSVs | RPS margin is derived from saved logits, not prose. |
| 4A | READY | Balanced sampling is the dominant adaptation factor for C4 MAE in one frozen-RPS OOF factorial | Phase 3.19 `summary/summary.json` | One backbone seed; not a severity result. |
| 4B | READY | The same factorial separates `z4-z3` margins | Phase 3.19 `metrics/margins.csv` | One backbone seed; no causal continuity with 4C--E. |
| 4C | READY | Lower support weakens mean p4 | Phase 3.20A per-seed and mean/SD CSVs | Full CE retraining; five seeds per support. |
| 4D | READY | Lower support weakens mean `z4-z3` | same | Show all five traces; no smoothing. |
| 4E | READY | Shrinkage is non-monotonic/seed-variable | same | Show all five traces; no smoothing. |

## Figure 1 — original CE phenomenon

All Figure 1 panels are frozen to the original, pre-intervention CE seed-0
model and exact discrete L1 Bayes decision. CE/RPS robustness is exclusively a
Figure 3 result.

| Panel / dataset | Exact artifact path(s) | Fields and deterministic work | IDs and alignment | Status / caveat |
| --- | --- | --- | --- | --- |
| 1A/1B/1C RetinaMNIST CE | `outputs/retinamnist/resolution_sanity_check/seed_0/size_28/predictions.csv`; `classwise_metrics.csv`; `config.json` | `sample_id`, `true_label`, `logits`, `probabilities`, stored predictive mean. Parse probabilities; compute exact L1 action, `mu=sum(k*p_k)`, `S=4-mu`, routing, and endpoint MAE. | Integer test IDs; one original seed-0 test population (`n=400`, C4=`20`). | READY_WITH_TRANSFORMATION. The stored `predicted_label` is nominal/mode, not the Figure 1 L1 action. |
| 1A/1B/1C UTKFace CE | `outputs/utkface/phase3_7a_failure_replication/ce/seed_0/predictions.csv`; `endpoint_routing.csv`; `endpoint_metrics.csv`; `metrics.json`; `config.json` | IDs, labels, logits, probabilities, and `l1_bayes_decision` are stored. Derive `mu` and endpoint shrinkage. | Stable string IDs; aligned test population `n=2,371`, C4=`67`. | READY_WITH_TRANSFORMATION. This is the original CE seed-0 matched replication, not a direction intervention. |
| 1A/1B/1C Solar CE | `outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/evaluation/predictions.npz`; `metrics.json`; `../config.json`; `../alignment_audit.json` | NPZ stores `sample_ids`, `labels`, `logits`, `probabilities`, `mode`, `l1`, `l2`, and risk. Derive `mu`/`S`; metrics directly contain class-0/X L1 MAE. | Integer IDs; aligned test `n=28,006`, X=`921`; unique IDs must be asserted. | READY_WITH_TRANSFORMATION. Retained aligned future-period test; Table 1/caption must retain temporal-overlap and retained-alignment caveat. |

## Figure 2 — CE representation → original CE head → CE direction

The panel is a CE-only mechanism decomposition. It does not combine the
Phase 3.3 Retina test population with Phase 3.18A OOF cases, and it does not
claim that a representation-inward sample is intrinsically unrecoverable.

| Panel | Dataset / conditions | Exact artifact path(s) | Population, fields, and alignment | Status / caveat |
| --- | --- | --- | --- | --- |
| 2A | Retina CE centroid strata | `outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/ce/seed_0/features.npz`; `outputs/retinamnist/phase3_18a_ce_direction_robustness/subgroups/class4_centroid_groups.csv` | NPZ stores `train_sample_id`, `train_labels`, `train_features`; compute raw train centroids and nearest centroid for true C4. Expected 48 nearest-4 and 18 representation-inward. Aggregate CSV is an independent saved check. | READY_WITH_TRANSFORMATION. Derive a per-ID group table before joining A/C. |
| 2B/2C/2D | Retina original A → CE C | `outputs/retinamnist/phase3_18a_ce_direction_robustness/predictions/A_CE_original.csv`; `C_CE_direction_only.csv`; `metrics/margins.csv`; `summary/results.json` | Both prediction CSVs store `sample_id`, `fold`, `label`, logits, probabilities, L1 action, and risk. Direct audit: 1,080 unique equal IDs and labels. Join their true-C4 rows to 2A groups. Expected exact A→C: 0→11, all 11 nearest-4, none among 18 inward. | READY_WITH_TRANSFORMATION. Recompute `mu`, `S`, exact flags and deltas from per-ID probabilities. |
| 2A | Solar CE centroid strata | `outputs/solar/phase3_9_mechanism_audit/features/ce/train.npz`; `test.npz`; `geometry/geometry.json` | `sample_ids`, labels, features, logits, probabilities and decisions are stored. Recompute Phase 3.9 raw train centroids and true-X nearest assignment. Expected X-like=`724`, inward=`197`. | READY_WITH_TRANSFORMATION. Assert IDs/labels against 3.18B before joining. |
| 2B/2C/2D | Solar original A → CE C | `outputs/solar/phase3_18b_ce_direction_robustness/evaluation/A_original_ce/predictions.npz`; `evaluation/C_direction_only/predictions.npz`; `x_representation_subgroups.csv`; `x_logit_margins.csv`; `summary.json` | A/C NPZs store IDs, labels, logits, probabilities, and L1 decisions. Saved subgroup check: X-like A `0/0/9/715/0` → C `0/0/49/116/559` (exact 0→559); inward A `0/23/51/123/0` → C `0/23/83/91/0` (0→0). | READY_WITH_TRANSFORMATION. The recovery concentration is descriptive; it is not an irrecoverability conclusion. |

## Figure 3 — direction response across objectives and domains

Use four within-setting A→C comparisons: Retina RPS, Retina CE, Solar RPS,
Solar CE. Panel B must plot exact **fraction**; `0→2`, `0→11`, `0→496/921`,
and `0→559/921` are annotations, not four directly comparable counts. Retina
is training-only five-fold OOF head evaluation; Solar is the predeclared
archived confirmatory readout. The comparison concerns sign and qualitative
consistency, not four equivalent independent test-set replications.

| Setting | Exact artifact path(s) | Population / fields | Required transformation and status |
| --- | --- | --- | --- |
| Retina RPS A/C | `outputs/retinamnist/phase3_10a_rop_objective_falsification/oof_predictions/A_original_rps.csv`; `outputs/retinamnist/phase3_10c_direction_only_head/oof_predictions/predictions.csv`; `summary/summary.json`; `margins/condition_c.csv` | Training-only OOF, seed 0, `n=1,080`, C4=`66`; CSVs provide IDs, labels, logits, probabilities, L1 action. | READY_WITH_TRANSFORMATION. Assert A/C C4 IDs/labels; derive A `z4-z3` from A logits and read/verify C margin from its CSV/logits. Known recovery 0→2 and shrinkage 1.797→1.384. |
| Retina CE A/C | `outputs/retinamnist/phase3_18a_ce_direction_robustness/predictions/A_CE_original.csv`; `C_CE_direction_only.csv`; `metrics/margins.csv`; `summary/results.json` | Training-only OOF, seed 0, `n=1,080`, C4=`66`; audited A/C IDs and labels match. | READY_WITH_TRANSFORMATION. Derive all summary fields from the paired CSVs. Saved CE margin is -0.903→+0.261; recovery 0→11; shrinkage 1.678→1.244. |
| Solar RPS A/C | `outputs/solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24/evaluation/A_original_rps/{predictions.npz,metrics.json}`; `evaluation/C_direction_only/{predictions.npz,metrics.json}`; `summary.json`; `split_integrity.json` | Archived seed-0 aligned test, `n=28,006`, X=`921`; NPZ fields include IDs, labels, logits, probabilities, L1 decisions, risks. | READY_WITH_TRANSFORMATION. Derive `zX-zM` from logits and exact fraction; known recovery 0→496/921 and shrinkage 1.228→.773. |
| Solar CE A/C | `outputs/solar/phase3_18b_ce_direction_robustness/evaluation/A_original_ce/{predictions.npz,metrics.json}`; `evaluation/C_direction_only/{predictions.npz,metrics.json}`; `x_logit_margins.csv`; `summary.json`; `split_integrity.json` | Archived seed-0 aligned test, `n=28,006`, X=`921`; same NPZ field contract. | READY_WITH_TRANSFORMATION. Margin CSV verifies -3.1086407→+1.1899806; known recovery 0→559/921 and shrinkage 1.136→.656. |

## Figure 4 — two explicitly separate regimes

**Block A — Phase 3.19 mechanism intervention study:** frozen Retina RPS
representation, direction-only heads, training-only OOF, one backbone seed,
natural/balanced × CE/RPS factorial. **Block B — Phase 3.20A controlled
severity study:** full CE backbone-plus-head retraining, five supports × five
seeds, validation checkpoint selection, and final predeclared test evaluation.
They must never be drawn as one causal trajectory. In particular, the
Phase 3.18A frozen-head OOF CE A baseline (C4 MAE about 1.439) is not
interchangeable with Phase 3.20A full-model `N4=66` baseline (2.330 ± .239).

| Panel | Exact artifact path(s) | Fields, IDs, and verification | Status / caveat |
| --- | --- | --- | --- |
| 4A Block A C4 MAE | `outputs/retinamnist/phase3_19_sampling_objective_direction_disentanglement/summary/summary.json`; `metrics/factorial_contrasts.csv` | C/F/E/G conditions, pooled OOF C4=`66`; source values E=1.697, G=1.712, C=1.348, F=1.318. | READY. One frozen RPS backbone seed; heading/caption: “Balanced sampling is the dominant adaptation factor.” |
| 4B Block A `z4-z3` | `outputs/retinamnist/phase3_19_sampling_objective_direction_disentanglement/metrics/margins.csv`; `predictions/oof_predictions.csv`; `parameters/direction_cosines.csv`; `subgroups/feature_nearest_c4.csv`; `metadata/provenance.json` | Margin CSV contains mean/median/positive fraction; prediction CSV contains IDs, labels, logits, probabilities and decisions. Direction cosine requires documented per-fold/class aggregation. | READY. Source means C=.030, E=-.912, F=.083, G=-.878; optional routing/shrinkage/p4/cosine appendix fields are available. |
| 4C Block B mean p4 | `outputs/retinamnist/phase3_20a_imbalance_severity_dose_response/analysis/per_seed_severity_metrics.csv`; `severity_mean_std.csv`; `completeness.json` | Per-seed CSV contains seed, support, p4, all endpoints/control metrics; mean/SD CSV is the saved aggregation. All 25 unique cells verified. | READY. Plot five seed traces plus saved mean; no smoothing. |
| 4D Block B mean `z4-z3` | same as 4C; each `n4_*/seed_*/{summary.json,config.json,manifest.json,evaluation.npz}` | Per-seed mean margin and positive fraction available; run files preserve validation-selected checkpoint provenance. | READY. Plot five traces plus saved mean; no smoothing. |
| 4E Block B shrinkage | same as 4C | Per-seed shrinkage and saved mean/SD available. | READY. Preserve non-monotonicity and seed variation; do not present a dose law. |

## Main tables

| Table | Machine-readable sources | Status and required caveat |
| --- | --- | --- |
| Table 1 — dataset/protocol | Retina `outputs/retinamnist/resolution_sanity_check/seed_0/size_28/config.json`; UTK `outputs/utkface/phase3_7a_failure_replication/dataset_audit/{dataset_audit.json,split_class_counts.csv}` and CE `config.json`; Solar Phase 3.8 CE `config.json`, `alignment_audit.json`, `metrics.json`; phase notes only for concise canonical-backbone prose. | READY_WITH_TRANSFORMATION. Keep native Retina setup, frozen historical UTK manifest, and Solar retained-aligned future-period/temporal-overlap caveat. |
| Table 2 — Figure 3 A/C direction summary | The four Figure 3 source pairs and their saved `metrics.json`/summary JSON/CSV files. | READY_WITH_TRANSFORMATION. Report within-setting rare MAE, exact fraction with count annotation, shrinkage, and global L1 MAE without crossing OOF and archived-test protocols. |

## Missing-data and transformation register

### Fully supported locally

- Original CE phenomenon inputs for RetinaMNIST, UTKFace, and Solar.
- CE mechanism-path inputs for Retina OOF and Solar aligned test.
- All four Figure 3 A/C settings, including saved logits for margins.
- Phase 3.19 factorial and Phase 3.20A 25-cell severity grid.

### Deterministic transformations required before plotting

- Parse serialized probability/logit vectors; compute exact discrete L1 actions
  where not stored, predictive mean, shrinkage, endpoint MAE, exact fractions,
  and margins.
- Recompute raw training centroids and the Figure 2 per-ID group label, then
  perform ID-keyed A/C joins with uniqueness and label assertions.
- Aggregate the Phase 3.19 saved per-fold class-direction cosine only under a
  declared aggregation rule; use saved Phase 3.20A mean/SD only after checking
  it against the 25 per-seed rows.

### Note-only or blocked sources

None for the proposed Figure 1--4 panels. Phase notes remain scientific
interpretation and cross-checks, not extraction sources. Do not promote
unsupported historical exploratory plots into the main figures merely because a
note reports an aggregate.

## Extraction stop conditions

Stop data consolidation and report the conflict if any ID set, label set,
split/protocol field, checkpoint/objective lineage, rare support, or saved
aggregate disagrees with this audit. Deterministic extraction is authorized by
this plan; new experiments, new metric selection, retraining, validation/test
reuse beyond the saved artifacts, and final figures are not.
