# Phase 3.19 — RetinaMNIST Sampling vs Objective Disentanglement

## Decision

\[
\boxed{\text{A — BALANCED SAMPLING IS THE PRIMARY DIRECTION-ADAPTATION DRIVER}}
\]

On the frozen RetinaMNIST RPS representation, balanced sampling was the dominant factor associated with beneficial classifier-direction adaptation. This is a training-only, one-backbone-seed RetinaMNIST mechanism result; it does not establish universal causality, select a method, or authorize a new run.

## Frozen factorial protocol

Only the established 1,080 canonical RPS training features, saved deterministic five OOF folds, and original RPS head were used. No validation or test arrays were loaded. Every adapted cell trained only class directions with original row norms and biases fixed for 100 epochs using AdamW (lr `.001`, zero weight decay, batch size `64`). C was validly reused from Phase 3.10C and its saved held-out logits replayed exactly.

| Sampling | CE | RPS |
| --- | --- | --- |
| Natural empirical | E | G |
| Established replacement-balanced | C | F |

F/G use the repository cumulative-probability `rps_loss`. Natural batches are shuffled fitting-fold passes; balanced batches use the established replacement sampler. All 20 adapted fold states retained original norms within `5.96e-8` and fixed biases exactly. Initial original-head replay error was `9.54e-7`; C's saved OOF replay error was `0`. OOF IDs/folds and the Phase 3.3 feature archive aligned exactly.

## Rare class-4 results

All routing is exact discrete L1 Bayes routing on pooled OOF probabilities.

| Condition | C4 routing `0/1/2/3/4` | Exact | C4 MAE | Mean p4 | Predictive mean | Shrinkage | Mean `z4-z3` / positive | C0 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A original RPS | `1/1/41/23/0` | 0 | 1.697 | .116 | 2.203 | 1.797 | -1.068 / 0.0% | .570 |
| C balanced + CE | `0/3/19/42/2` | 2 | 1.348 | .306 | 2.616 | 1.384 | .030 / 57.6% | .691 |
| E natural + CE | `1/4/35/26/0` | 0 | 1.697 | .141 | 2.157 | 1.843 | -.912 / 0.0% | .506 |
| F balanced + RPS | `0/2/19/43/2` | 2 | 1.318 | .313 | 2.637 | 1.363 | .083 / 63.6% | .658 |
| G natural + RPS | `2/6/29/29/0` | 0 | 1.712 | .141 | 2.153 | 1.847 | -.878 / 0.0% | .498 |

Balanced sampling produces the same substantial outward change under both objectives: C versus E lowers C4 MAE by `.348`, reduces shrinkage by `.459`, and raises the mean `z4-z3` margin by `.942`; F versus G changes those values by `.394`, `.484`, and `.960`. CE-versus-RPS differences within balanced sampling are small (C minus F C4 MAE `+.030`, shrinkage `+.021`, margin `-.053`), as are objective differences under natural sampling (MAE `-.015`, shrinkage `-.004`, margin `-.035`).

Mean A-to-final C4 cosine was `.430` for C and `.432` for F, versus `.497` for E and `.495` for G; the stronger balanced rotations coincide with positive margins and outward localization. This is association within the controlled factorial protocol, not causal proof beyond it.

Among 42 feature-nearest-4 cases, C/F route `42/41` outward (>=3) and recover two exact cases each; E/G route only `26/29` outward and recover none. No condition exactly recovers the 24 representation-inward cases. Centroid status remains descriptive, not causal proof.

## Questions answered

1. **Q1, C vs E:** balanced sampling improves direction-only C4 localization under CE.
2. **Q2, F vs G:** the same sampling effect occurs under RPS.
3. **Q3–Q4:** CE has no meaningful advantage within balanced or natural sampling.
4. **Q5–Q7:** evidence is sampling-dominant, not an interaction; direction/margin/localization move together.
5. **Q8:** the manuscript may state this bounded RetinaMNIST/RPS-representation training-signal result, while remaining agnostic outside this protocol.

Global and risk/UQ values are diagnostic, not selection criteria. Both balanced cells carry class-0 and probability/global trade-offs relative to natural cells; rare-end localization is not a global or UQ gain claim.

## Scope and stop condition

No Solar or CE-representation factorial, backbone retraining, scale/bias intervention, ROP, hyperparameter tuning, seed expansion, validation/test access, commit, or push occurred. Artifacts are preserved in `outputs/retinamnist/phase3_19_sampling_objective_direction_disentanglement/`.
