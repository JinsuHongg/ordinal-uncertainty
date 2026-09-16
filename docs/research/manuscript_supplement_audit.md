# Manuscript Supplement Audit

**Status:** Solar provenance recovery updated (2026-09-15)  
**Scope:** reproducibility/provenance and supplemental results only; no new
experiment, analysis, figure, claim, commit, or push.

## Sections and tables created

- S1 reproducibility, consolidated configuration, and Solar provenance
  (`reproducibility.tex`)
- S2 confirmatory seed-level H1 and retained H2a results
  (`confirmatory_seed_results.tex`)
- S3 historical seed-0 mechanism evidence, B/D context, Phase 3.19, and
  Phase 3.20A (`historical_mechanism.tex`)
- S4 UTKFace historical supporting evidence (`utkface_supporting.tex`)
- S5 H2b and global/UQ context (`secondary_diagnostics.tex`)

Created tables cover configuration, Solar provenance, 16 seed-level H1 rows,
eight retained Retina seed-level H2a rows, the frozen H2a verdict matrix,
historical centroid strata, Phase 3.19, Phase 3.20A, UTKFace, and Retina H2b.

## Provenance outcome

The supplement closes the documentation gap for the frozen A/C head protocol,
Retina OOF/centroid provenance, and the retained Solar channel, transform,
normalization, archived-readout, checkpoint-lineage, and test-selection rules.

One artifact-availability gap remains and is explicitly disclosed rather than
filled by inference:

1. Solar seed-level H2a regression outputs (coefficients, HC3 intervals, and
   LOO deltas) are not locally recovered. The supplement reports only the
   frozen four-setting verdict matrix from the cross-setting synthesis.

The recovered provenance index records all eight exact archived checkpoint
paths, selected epochs, validation losses, READY status, and protocol-level
configuration. The retained Solar alignment audit verifies aligned
train/validation/test counts of 45,047/2,431/28,006.

## Reproducibility checklist

All reviewer questions are answerable except where explicitly marked as an
artifact gap: exact backbone configuration is complete for Retina and specified
by the Solar frozen runner; seed roles, C fitting data, centroid fitting data,
evaluation roles, test/model-selection exclusion, balanced sampling, C
hyperparameters, H1/H2a gates, Solar readout role, and UTKFace exclusion are
all stated. Historical and confirmatory results are separated throughout.

## Main-text and claim impact

No main-text source file was modified and no scientific claim changed. The
supplement adds no figures or literature. Historical B/D, UTKFace, H2b, and
global/UQ metrics remain supplement-only context.

## Recommended next step

Before submission, recover/archive the saved Solar seed-level H2a analysis
outputs. Do not regenerate them by rerunning the analysis.
