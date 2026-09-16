# Solar Provenance Recovery

**Status:** **A — SOLAR PROVENANCE RECOVERY COMPLETE** (2026-09-15)  
**Scope:** local existing-artifact recovery only. No backbone training, A/C
replay, H1/H2 analysis, H2a recomputation, output regeneration, commit, or push
was performed.

## 1. Executive verdict

The Solar backbone/configuration provenance gap is **CLOSED** under the frozen
recovery criterion: all eight confirmatory conditions have an exact archived
checkpoint path plus verified per-seed selection metadata and a
protocol-level-verified configuration source. Aligned train/validation/test
counts are also **CLOSED** from a retained alignment artifact.

The Solar seed-level H2a regression outputs are also **CLOSED**. The existing
CE and RPS saved per-seed tables contain \(\beta_{\rm ord}\), HC3 intervals,
LOO MSE values, and \(\Delta\mathrm{LOO}\) for all four confirmatory seeds.
Their frozen verdicts are unchanged.

## 2. Search scope

Read-only searches covered the local repository output trees, research
provenance notes, execution logs, frozen runner, and accessible mounted Solar
artifact area. In particular:

- `outputs/solar/mechanism_replication/`
- `outputs/mechanism_replication/ac/solar/`
- `outputs/mechanism_replication/analysis/solar/`
- `outputs/solar/phase3_8_shrinkage_confirmation/`
- `docs/research/{stage1_mechanism_provenance_audit,solar_replication_execution_log,solar_ac_replication_execution_log}.md`
- `scripts/train_solar_replication_backbone.py`

The local `outputs/mechanism_replication/analysis/solar/` tree contains the
saved CE/RPS H2a analysis files. This recovery audit verifies those stored
outputs only; it does not re-fit any regression from the A/C per-sample rows.

## 3. Backbone manifest recovery

The canonical provenance index is:

`outputs/solar/mechanism_replication/provenance/solar_confirmatory_manifest.json`

For CE and RPS seeds 1--4, it records the exact archived checkpoint path,
selected epoch, validation loss, selection criterion, A/C manifest lineage,
and per-seed integrity status. It separately identifies common training values
as **protocol-level verified**, sourced from the frozen runner and Stage 1
provenance audit:

- unpretrained torchvision ResNet18 with a `512 -> 5` head;
- AdamW, learning rate `5e-5`, weight decay `.01`, batch size `16`;
- 300 maximum epochs, patience `3`;
- independent horizontal and vertical flips;
- channels `hmi_m/aia1600/aia131` with indices `[8,7,1]`;
- signed-log transform and train-only normalization.

The per-seed `config.json`, training history, alignment audit, and selected
checkpoint are now retained locally for every condition. The backbone portion
is therefore fully locally recovered rather than only path/metadata recovered.

## 4. Checkpoint provenance

| Objective | Seed | Epoch | Validation loss | Checkpoint status |
| --- | ---: | ---: | ---: | --- |
| CE | 1 | 1 | 1.221391 | LOCALLY RECOVERED |
| CE | 2 | 1 | 1.037971 | LOCALLY RECOVERED |
| CE | 3 | 1 | 1.096287 | LOCALLY RECOVERED |
| CE | 4 | 1 | 1.155523 | LOCALLY RECOVERED |
| RPS | 1 | 2 | .098172 | LOCALLY RECOVERED |
| RPS | 2 | 1 | .082588 | LOCALLY RECOVERED |
| RPS | 3 | 1 | .086884 | LOCALLY RECOVERED |
| RPS | 4 | 1 | .091880 | LOCALLY RECOVERED |

Sources are `outputs/solar/mechanism_replication/backbone_integrity_audit.json`
and the matching A/C `manifest.json` under
`outputs/mechanism_replication/ac/solar/`.

## 5. H2a seed-level artifact recovery

**CLOSED.** The recovered existing results are:

- `outputs/mechanism_replication/analysis/solar/{ce,rps}/h2a_per_seed.csv`;
- matching `analysis_manifest.json` and `h2a_summary.json`; and
- `outputs/mechanism_replication/analysis/solar/solar_confirmatory_summary.json`.

Both objective-specific manifests identify the four independent backbone seeds,
true class-4 rows, sample-ddof=1 within-seed standardization, HC3 OLS, and
training-fold-only LOOCV standardization. The CE table reports coefficients
`[-.219, -.367, -.328, -.146]`; the RPS table reports
`[-.324, -.345, -.310, -.046]`. All eight coefficients are negative and all
eight stored \(\Delta\mathrm{LOO}\) values are positive. The stored summaries
therefore retain the frozen **STRONG** verdict for both CE and RPS.

`outputs/solar/mechanism_replication/provenance/solar_h2a_artifact_index.json`
indexes the two per-seed tables, manifests, and summaries with SHA-256 checksums.
It records source-artifact identity only; no raw A/C row was reprocessed.

## 6. Aligned count recovery

**CLOSED.** `outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/alignment_audit.json`
records aligned counts of 45,047 train, 2,431 validation, and 28,006 test.
The test X support is 921. The same counts are independently retained in the
Phase 3.18B CE split-integrity artifact.

## 7. Files created

- `outputs/solar/mechanism_replication/provenance/solar_confirmatory_manifest.json`
- `outputs/solar/mechanism_replication/provenance/solar_h2a_artifact_index.json`
- `docs/research/solar_provenance_recovery.md`

## 8. Supplement updates

`manuscript/supplement/sections/reproducibility.tex` already reports the
confirmatory seed-level H2a results. The recovered source tables now provide
direct provenance for that material; no manuscript claim or value changed.

## 9. Remaining gaps

| Original gap | Recovery state | Consequence |
| --- | --- | --- |
| Per-seed backbone configuration/checkpoint manifests | **CLOSED** | Exact paths and selection metadata plus protocol-level configuration are preserved. |
| Seed-level Solar H2a outputs | **CLOSED** | Existing saved per-seed regression outputs and checksummed index are retained. |
| Aligned train/validation counts | **CLOSED** | Retained alignment audit verifies all three split counts. |

## 10. Submission readiness

The prior Solar provenance blocker is closed. Venue formatting remains a
separate task and is not authorized by this recovery audit.
