# Natural-Sampling Direction-Only Pre-Execution Audit

**Status:** RetinaMNIST artifact blocker resolved on 2026-09-16; no N fits were run.

## Scope and stop rule

This audit covers the user-authorized N condition: a natural-empirical-sampling,
direction-only head fit paired with each frozen confirmatory A/C setting.  It
does not rerun inference, train a backbone, regenerate Solar features, create
N output trees, or update manuscript claims.

The execution request requires a locally saved frozen feature artifact for each
setting before fitting N.  A separately authorized Retina-only deterministic
feature-regeneration task has now recovered the eight RetinaMNIST archives;
see `retina_confirmatory_frozen_feature_regeneration.md`. Solar remains an
explicit stop condition and is reserved for separate cluster work.

## Artifact audit

For every row below, `manifest.json`, `per_sample_arrays.npz`, the original A
head (`A_original_head.pt`), and the corresponding C head state are present in
`outputs/mechanism_replication/ac/<dataset>/<objective>/seed_<seed>/` (five
out-of-fold C states for RetinaMNIST; one C state for Solar).  The
per-sample arrays contain the archived labels, sample IDs, folds, A/C logits,
probabilities, decisions, and diagnostics, but no 512-D feature matrix.  No
matching saved train feature matrix was found under the confirmatory output
tree.

| Dataset | Objective | Seed | Frozen checkpoint recorded | Saved matching features | Result |
|---|---:|---:|---|---|---|
| RetinaMNIST | CE | 1 | yes | yes | ready |
| RetinaMNIST | CE | 2 | yes | yes | ready |
| RetinaMNIST | CE | 3 | yes | yes | ready |
| RetinaMNIST | CE | 4 | yes | yes | ready |
| RetinaMNIST | RPS | 1 | yes | yes | ready |
| RetinaMNIST | RPS | 2 | yes | yes | ready |
| RetinaMNIST | RPS | 3 | yes | yes | ready |
| RetinaMNIST | RPS | 4 | yes | yes | ready |
| Solar | CE | 1 | yes | no | blocked: feature regeneration required |
| Solar | CE | 2 | yes | no | blocked: feature regeneration required |
| Solar | CE | 3 | yes | no | blocked: feature regeneration required |
| Solar | CE | 4 | yes | no | blocked: feature regeneration required |
| Solar | RPS | 1 | yes | no | blocked: feature regeneration required |
| Solar | RPS | 2 | yes | no | blocked: feature regeneration required |
| Solar | RPS | 3 | yes | no | blocked: feature regeneration required |
| Solar | RPS | 4 | yes | no | blocked: feature regeneration required |

Historical RetinaMNIST seed-0 or earlier-phase archives were not substituted.
The regenerated seed-1--4 archives are bound to their exact confirmatory
checkpoints and archived A/C fold assignments. No Solar train feature archives
were found in the mechanism-replication output tree.

## Verified reusable protocol metadata

The 16 A/C manifests consistently specify the frozen 512-D input to `model.fc`
after average pooling/flattening.  Condition C used cross-entropy, replacement
class-balanced sampling, AdamW (`lr=0.001`, weight decay `0`), batch size 64,
100 epochs, fixed original class-wise norms and biases, and the archived
training-only evaluation protocol.  The archived A/C per-sample artifacts also
provide the intended evaluation populations and A/C endpoint summaries, but
they are insufficient to refit a head because logits are not features.

## Exact blocker and required recovery

N cannot be fit or compared with C until a feature artifact is available for
each target setting with, at minimum:

1. train features, labels, and stable sample IDs for the original head-fitting
   partition (including the original fold assignments where applicable);
2. readout/evaluation features, labels, and sample IDs aligned to the archived
   A/C population;
3. provenance binding both matrices to the exact frozen checkpoint and seed;
4. the original-head weights, biases, and class-wise norms used for the A/C
   replay.

All four requirements now hold for RetinaMNIST CE/RPS seeds 1–4. Retina Phase B
is technically ready but remains unrun pending separate authorization. For
Solar specifically, obtaining items 1–2 currently requires frozen-backbone
feature regeneration. Solar N must remain unrun unless those saved artifacts
are recovered or a separately scoped cluster procedure is authorized.
