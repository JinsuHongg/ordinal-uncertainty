# Paper Figure Source-Artifact Audit

## Scope and verdict

**Audit date:** 2026-09-10. This is a source-artifact availability and provenance audit only: no models were run, no outputs were regenerated, and no figures, plotting scripts, or final figure-data tables were created.

**Verdict: C — MATERIAL FIGURE-SOURCE GAPS REQUIRE REVIEW.**

Complete machine-readable Solar CE/RPS baseline and direction-head artifacts are present. The expected RetinaMNIST and UTKFace output trees for Figure 1–4 are absent from this checkout. Therefore the cross-dataset panels, Retina direction cells, and both Figure 4 blocks are not reproducible locally from machine-readable artifacts. Phase-note summaries are evidence, not substitute extraction sources.

Status: **READY** = all fields present; **READY_WITH_TRANSFORMATION** = an ID-keyed join or documented deterministic derivation is needed; **PARTIALLY_SUPPORTED** = only part of a proposed panel is locally available; **BLOCKED** = no safe local machine-readable source.

## Audit method and constraint

The audit reviewed only the requested phase-note output trees, listed their JSON/CSV/NPZ files, and inspected NPZ member schemas without loading full feature tensors. The supplied Python runtime lacks NumPy, so byte-level NPZ array equality was not independently recomputed. Future Figure 2 extraction must explicitly assert unique IDs, identical ID sets, and label equality across centroid, A, and C sources before emitting a table. No positional alignment is permitted.

## Source matrix

| Panel | Status | Primary source | Population | Main caveat |
|---|---|---|---|---|
| Fig. 1A | PARTIALLY_SUPPORTED | Solar CE Phase 3.8 predictions | CE rare endpoint, seed 0 | Retina/UTK artifact trees absent. |
| Fig. 1B | PARTIALLY_SUPPORTED | Solar CE Phase 3.8 predictions | CE rare endpoint, seed 0 | Retina/UTK artifact trees absent. |
| Fig. 1C | PARTIALLY_SUPPORTED | Solar CE Phase 3.8 metrics | CE class 0 and rare endpoint | Retina/UTK artifact trees absent. |
| Fig. 2A | PARTIALLY_SUPPORTED | Solar CE Phase 3.9 features | Solar X=921; Retina C4=66 OOF | Solar derivation ready; Retina absent. |
| Fig. 2B | PARTIALLY_SUPPORTED | Solar 3.9 + 3.18B CE | same as Fig. 2A | Requires ID-keyed join; Retina absent. |
| Fig. 2C | PARTIALLY_SUPPORTED | Solar 3.18B CE subgroup/A/C | same as Fig. 2A | Solar aggregate provenance present; Retina absent. |
| Fig. 2D | PARTIALLY_SUPPORTED | Solar 3.9 + 3.18B CE | same as Fig. 2A | Derive A/C delta; Retina absent. |
| Fig. 3A | PARTIALLY_SUPPORTED | Solar 3.15 RPS + 3.18B CE | A/C, four settings | Both Retina cells absent. |
| Fig. 3B | PARTIALLY_SUPPORTED | same | A/C, four settings | Use fractions; Retina denominator absent. |
| Fig. 3C | PARTIALLY_SUPPORTED | same | A/C, four settings | Both Retina cells absent. |
| Fig. 3D | PARTIALLY_SUPPORTED | Solar CE margin CSV/RPS logits | A/C, four settings | Both Retina margins absent. |
| Fig. 4A | BLOCKED | expected Phase 3.19 outputs absent | Retina RPS OOF C/F/E/G | Note-only values are insufficient. |
| Fig. 4B | BLOCKED | expected Phase 3.19 outputs absent | Retina RPS OOF C/F/E/G | Note-only values are insufficient. |
| Fig. 4C | BLOCKED | expected Phase 3.20A outputs absent | 5 supports × 5 seeds | 25 cells cannot be audited. |
| Fig. 4D | BLOCKED | expected Phase 3.20A outputs absent | 5 supports × 5 seeds | 25 cells cannot be audited. |
| Fig. 4E | BLOCKED | expected Phase 3.20A outputs absent | 5 supports × 5 seeds | 25 cells cannot be audited. |

## Figure 1 — original CE phenomenon baseline

Use original pre-intervention CE, seed 0, and exact discrete L1 Bayes decision. Do not choose the objective with the largest failure.

| Dataset / panel | Required fields | Exact source / expected source | Support and status |
|---|---|---|---|
| Solar CE 1A/1B | `sample_ids`, `labels`, `l1`, `probabilities` | `outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/evaluation/predictions.npz` | X=921; READY for routing, READY_WITH_TRANSFORMATION for `mu=sum(k*p_k)` and `S=4-mu`. |
| Solar CE 1C | class-0 and X L1 MAE | `outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/evaluation/metrics.json` | class 0=5,284, X=921; READY. |
| Retina CE 1A–C | IDs, labels, L1, probabilities/metrics | Expected Phase 3.3 CE lineage `outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/ce/seed_0/features.npz` plus referenced CE prediction artifact; absent | Phase note has test C4=20; BLOCKED. |
| UTKFace CE 1A–C | same | Expected `outputs/utkface/phase3_7a_failure_replication/`; absent | Phase note has test C4=67; BLOCKED. |

Solar is a retained alignment subset (28,006 aligned test rows, X=921) and uses a future-period test with temporal overlap in train/validation; retain this Table 1 and Figure 1 caveat.

## Figure 2 — frozen CE mechanism decomposition

Retina must use Phase 3.18A training-only OOF CE (`n=1,080`, C4=66), never the Phase 3.3 test C4=20. Solar uses Phase 3.9 CE raw train centroids and the Phase 3.18B archived CE A/C readout (`n=28,006`, X=921).

| Panel | Retina CE source/status | Solar CE source/status | Required transformation/caveat |
|---|---|---|---|
| 2A representation state | Expected Phase 3.18A plus Phase 3.3 CE feature source; absent, BLOCKED | `outputs/solar/phase3_9_mechanism_audit/features/ce/{train,test}.npz`, READY_WITH_TRANSFORMATION | Compute raw train centroids, then group true rare-end IDs. Expected Solar 724 X-like/197 inward; Retina 48 nearest-4/18 inward. |
| 2B A routing by state | Expected Phase 3.18A A OOF predictions; absent, BLOCKED | Phase 3.9 CE test NPZ + `outputs/solar/phase3_18b_ce_direction_robustness/evaluation/A_original_ce/predictions.npz`, READY_WITH_TRANSFORMATION | Join by IDs, not position. |
| 2C A→C exact recovery | Expected Phase 3.18A A/C OOF predictions; absent, BLOCKED | Phase 3.9 CE cache; Phase 3.18B A/C NPZ; `x_representation_subgroups.csv`, READY_WITH_TRANSFORMATION | CSV records Solar X-like 0→559 exact and inward 0→0; regenerate groups from centroids for sample-level table. |
| 2D A→C movement | Expected Phase 3.18A A/C probabilities; absent, BLOCKED | same Solar sources, READY_WITH_TRANSFORMATION | ID-keyed join; compute `S_C-S_A` or `mu_C-mu_A`. |

Solar subgroup provenance is explicit: X-like=724, A routing `0/0/9/715/0`, C `0/0/49/116/559`, MAE `1.012→.296`, exact `0→559`; inward=197, A `0/23/51/123/0`, C `0/23/83/91/0`, MAE `1.492→1.655`, exact `0→0`. All exact recoveries are X-like, descriptively—not proof that inward cases are intrinsically unrecoverable.

## Figure 3 — A→C direction response

RetinaMNIST is training-only OOF head evaluation; Solar is the predeclared archived confirmatory readout. Compare sign and qualitative consistency, not four equivalent independent test-set replications. Plot exact rare-end **fraction**, with raw counts as annotations only.

| Setting | Exact present source | Population / needed fields | Status |
|---|---|---|---|
| Retina RPS | Expected `outputs/retinamnist/phase3_10c_direction_only_head/`; absent | OOF seed 0; recover denominator, MAE, shrinkage, logits | BLOCKED. |
| Retina CE | Expected `outputs/retinamnist/phase3_18a_ce_direction_robustness/`; absent | OOF seed 0, C4=66; same fields | BLOCKED. |
| Solar RPS | `outputs/solar/phase3_15_direction_scale_mechanism_confirmation_retry3_qgpu24/evaluation/{A_original_rps,C_direction_only}/{metrics.json,predictions.npz}` and `summary.json` | archived seed 0, X=921 | READY_WITH_TRANSFORMATION; derive `zX-zM` from logits. |
| Solar CE | `outputs/solar/phase3_18b_ce_direction_robustness/evaluation/{A_original_ce,C_direction_only}/{metrics.json,predictions.npz}`, `summary.json`, `x_logit_margins.csv` | archived seed 0, X=921 | READY_WITH_TRANSFORMATION; margin CSV directly gives A -3.1086407, C 1.1899806. |

Solar CE/RPS NPZs contain IDs, labels, logits, probabilities, mode/L1/L2, and risks. Solar RPS has no separately located margin summary, but logits provide the deterministic source. Do not copy Retina note values into a future figure table; restore artifacts first.

## Figure 4 — two distinct experimental regimes

### Block A: Phase 3.19 mechanism intervention

Frozen Retina RPS features; direction-only heads; five-fold training-only OOF; one backbone seed; natural/balanced × CE/RPS. Expected directory `outputs/retinamnist/phase3_19_sampling_objective_direction_disentanglement/` is absent. Thus 4A (C4 MAE: E 1.697, G 1.712, C 1.348, F 1.318), 4B (mean `z4-z3`: C .030, E -.912, F .083, G -.878), and appendix candidates (shrinkage/routing/p4/positive fraction/cosine) are **BLOCKED**. These values are note-only in this checkout.

### Block B: Phase 3.20A controlled severity

Full CE backbone-plus-head retraining; supports `66,50,33,16,8` × five seeds; validation checkpoint selection; final predeclared test evaluation. Expected directory `outputs/retinamnist/phase3_20a_imbalance_severity_dose_response/` is absent. Therefore 4C mean p4, 4D mean `z4-z3`, and 4E shrinkage are all **BLOCKED**. No audit can verify 25 unique `(seed,n4_support)` cells, full seed/support coverage, or agreement with phase-summary aggregation. Do not connect its full-model `N4=66` baseline to Phase 3.18A frozen-head OOF CE.

## Future main tables

| Table | Sources | Status | Caveat |
|---|---|---|---|
| Table 1 dataset/protocol | Phase 3.3, 3.7A, 3.8 notes plus original configs/manifests | PARTIALLY_SUPPORTED | Solar source metadata is present; Retina/UTK source configs/artifacts are absent. Preserve Solar retained-subset and temporal-split caveat. |
| Table 2 direction summary | Phase 3.10C, 3.18A, 3.15, 3.18B A/C summaries/predictions | PARTIALLY_SUPPORTED | Solar CE/RPS present; Retina cells absent. Every value must remain within its setting's protocol. |

## Missing-data / unresolved-gap register

### Fully supported locally

- Solar CE Figure 1 routing, probability location, shrinkage, and endpoint MAE.
- Solar CE/RPS Figure 3 rare MAE, exact fraction, shrinkage, global L1 metrics, and logits/probabilities for deterministic recomputation.

### Deterministic transformations required

- Derive predictive mean/shrinkage and normalized exact fractions.
- Derive Solar CE Figure 2 centroid groups and ID-keyed A/C joins.
- Derive Solar RPS `zX-zM` from logits.

### Note-only or blocked in this checkout

- Retina CE/RPS Figure 2/3 sources; UTKFace CE Figure 1 source.
- Phase 3.19 factorial source files.
- Phase 3.20A per-run 25-cell severity files.

**Smallest remediation:** restore/copy the cited historic output directories and configs into a read-only accessible artifact location; then run only deterministic extraction, uniqueness, and provenance checks. This is artifact recovery, not a new experiment.
