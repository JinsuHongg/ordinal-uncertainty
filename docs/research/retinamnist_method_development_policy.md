# RetinaMNIST Method Development Policy

## Scope

This policy governs iterative RetinaMNIST method development in the
`ordinal-uncertainty` repository.

It centralizes the rules that should not be repeated in every Phase 3.10x
prompt.

Phase-specific instructions may override this document only when the override
is explicit.

---

## Canonical Development Setup

Use the canonical RetinaMNIST configuration:

- native **28×28 RGB**
- 5 ordinal classes: `0 < 1 < 2 < 3 < 4`
- official dataset identities
- unpretrained small-image ResNet18
- 3×3 stride-1 stem
- no max-pool

Canonical training class counts:

```text
[486, 128, 206, 194, 66]
```

Class 4 is the rare upper extreme.

Historical 64×64 results are sensitivity evidence only and must not replace the
native 28×28 canonical setup.

---

## Canonical RPS Development Reference

Unless a phase explicitly authorizes another representation, use the canonical
RetinaMNIST seed-0 RPS checkpoint and its frozen penultimate representation.

Current reusable development artifacts include:

- canonical seed-0 RPS checkpoint;
- frozen 512-D RPS penultimate features;
- canonical RPS logits/probabilities;
- Phase 3.10A deterministic stratified 5-fold assignments;
- Phase 3.10A/3.10B OOF outputs and head artifacts.

Verify artifact integrity before reuse.

Do not silently retrain a backbone or regenerate artifacts when exact reusable
artifacts already exist.

---

## Development Split Policy

RetinaMNIST has been inspected extensively.

Iterative method development must therefore use the **training set only**.

Reuse the exact Phase 3.10A deterministic stratified 5-fold OOF assignments
unless a later phase explicitly changes the protocol.

Requirements:

- all 1,080 training samples are retained;
- each sample appears in exactly one held-out fold;
- each sample appears in four fitting folds;
- stratification is preserved as closely as possible;
- held-out fold labels are evaluation-only;
- fold assignment is deterministic and saved.

Do not construct new splits simply because a phase result is inconvenient.

---

## Historical Validation/Test Policy

During iterative method development:

### Historical validation

Do not use historical validation values for:

- tuning;
- selecting among candidate variants;
- fold construction;
- post-hoc threshold selection.

Historical validation may only be used after an explicit method-freeze or
separate validation authorization.

### Historical test

The historical RetinaMNIST test set is prohibited for:

- iterative objective design;
- hyperparameter tuning;
- candidate selection;
- deciding which method variant to keep;
- debugging based on scientific performance.

Do not inspect test scientific metrics during an iterative development phase
unless explicitly authorized.

Historical test access requires a separate final-evaluation authorization.

---

## Development vs Confirmatory Evidence

RetinaMNIST is a **development benchmark**.

A promising training-only OOF result is development evidence only.

It does not establish cross-dataset validity.

After a method is explicitly frozen, the intended confirmatory progression is:

```text
RetinaMNIST final/frozen evaluation
    ↓
UTKFace
    ↓
Solar
```

Do not launch seeds 1–4 or additional datasets before explicit authorization.

---

## Frozen-Representation Discipline

If a phase is defined as a head-only or parameter-only experiment:

- the backbone must remain frozen;
- frozen features must not receive gradients;
- representation artifacts must not be mutated;
- any claimed head effect must be isolated from representation changes.

If representation is changed in a later phase, that change must be explicitly
authorized and controlled.

---

## Common Decision Rules

When probabilities are evaluated, report:

- mode;
- exact discrete L1-optimal decision;
- exact discrete L2-optimal decision;

when applicable to the phase.

The exact L1 decision remains the primary ordinal decision control.

---

## Common Global Metrics

Unless a phase explicitly narrows evaluation, report:

- Accuracy
- MAE
- QWK
- severe-error prevalence
- NLL
- Brier score
- RPS
- ECE

Use existing repository implementations.

---

## Common Ordinal Risk Metrics

Using established exact ordinal decision risk, report when applicable:

- L1-risk / error Spearman;
- severe-error AUROC;
- severe-error AUPRC;
- ordinal-MAE risk-coverage;
- mean selective MAE.

Do not invent a new uncertainty metric during a method-development phase unless
a separate literature/design task explicitly authorizes it.

---

## Rare Upper-Extreme Diagnostics

For true class 4, report when applicable:

- routing `4→0/1/2/3/4`
- exact recovery
- class-4 MAE
- severe prevalence
- mean/median `p4`
- mean/median `p3`
- mean/median `p3+p4`
- predictive mean
- inward shrinkage
- mean/median L1 Bayes risk

Define inward shrinkage as:

\[
4-\sum_{k=0}^{4} k p_k.
\]

Do not hard-code class-4-specific training corrections unless explicitly
authorized. Class-4 diagnostics are evaluation quantities, not automatic
training targets.

---

## Lower-Endpoint Safety Control

For true class 0, report when applicable:

- routing
- MAE
- severe prevalence
- `p0`
- `p1`
- `p0+p1`
- predictive mean
- L1 Bayes risk

Class 0 is the mandatory endpoint-safety control.

A method that improves class 4 by materially damaging class 0 is not a clean
success.

---

## Common Experimental Discipline

Every experiment should:

1. ask one narrow scientific question;
2. use predeclared conditions;
3. keep matched conditions identical except for the tested intervention;
4. predeclare GO / TRADE-OFF / NO-GO criteria when applicable;
5. avoid post-hoc hyperparameter expansion;
6. preserve valid negative results;
7. stop when the scientific question is answered;
8. avoid automatic branch expansion after an interesting intermediate result.

Do not create iterative variants from observed OOF or test failures unless a new
phase is explicitly authorized.

---

## Hyperparameter Discipline

Each new phase should introduce as few new scientific hyperparameters as
possible.

Rules:

- freeze the grid before viewing scientific comparison results;
- do not expand the grid post hoc;
- do not tune to class-4 exact recovery alone;
- do not introduce class-specific coefficients without explicit justification;
- prefer causal/mechanistic tests before flexible optimization.

---

## Artifact Requirements

For every new development condition, save enough information for exact replay.

When applicable save:

- sample ID;
- OOF fold;
- label;
- logits;
- probabilities;
- mode/L1/L2 decisions;
- L1 Bayes risk;
- condition name;
- hyperparameters;
- fitted head state;
- optimizer/training metadata.

Do not overwrite historical Phase 3.10A/B artifacts.

Use a new phase-specific output directory.

---

## Repository Safety

Do not:

- delete datasets or checkpoints automatically;
- overwrite historical outputs;
- rewrite valid negative-result artifacts;
- remove unrelated working-tree changes;
- retrain a backbone unless explicitly authorized;
- run seeds 1–4 unless explicitly authorized;
- run UTKFace or Solar unless explicitly authorized;
- commit or push unless explicitly requested.

If a required action conflicts with preserved research evidence, stop and
report the conflict.

---

## Verification

Before declaring a phase complete, run:

```bash
pytest -q
python -m compileall src scripts
git diff --check
```

Use the repository's verified active environment.

Current known usable environment:

```text
/mnt/storage/conda_envs/ordinal-uncertainty
```

Do not require the old `ocqr` alias if it is unavailable.

Also inspect:

```bash
git status --short
git diff --stat
```

Report failures explicitly.

---

## Documentation Policy

Every completed research phase must have a dedicated note under:

`docs/research/`

The phase-specific note is the detailed scientific source of truth.

After the phase decision is complete, review and synchronize as needed:

- `docs/research/current_state.md`
- `docs/research/decision_log.md`
- `docs/research/experiment_plan.md`
- `AGENTS.md`
- `README.md` only when high-level public-facing direction changes

Do not repeat detailed experimental logs in `README.md`.

---

## Documentation Hierarchy

When scientific-state documents disagree, use:

1. latest completed relevant phase-specific note;
2. `docs/research/decision_log.md`;
3. `docs/research/current_state.md`;
4. `docs/research/experiment_plan.md`;
5. `AGENTS.md`;
6. `README.md`.

After resolving a disagreement, update stale documents before finishing the
task.

---

## Phase Prompt Contract

Future Phase 3.10x prompts should normally include only:

1. task / phase name;
2. scientific question;
3. prior evidence needed for the phase;
4. new intervention;
5. conditions/comparators;
6. phase-specific diagnostics;
7. decision gate;
8. stop condition.

They should begin with language equivalent to:

> Read `AGENTS.md`, `docs/research/retinamnist_method_development_policy.md`,
> and the relevant completed phase notes. All common RetinaMNIST dataset, OOF,
> evaluation, safety, verification, artifact, and documentation rules are
> inherited from the shared policy unless explicitly overridden below.

Do not duplicate shared rules unless the phase needs an explicit exception.

---

## Current Phase-Family Guardrail

As of completion of Phase 3.10B:

\[
\boxed{\text{MIXED PARAMETER MECHANISM}}
\]

The balanced-head trade-off is mediated by classifier direction rotation and
class-dependent norm inflation; bias shifts were negligible.

The next authorized mechanism test is Phase 3.10C:

\[
\boxed{\text{Direction-Only Head Adaptation Falsification}}
\]

Phase 3.10C may isolate classifier direction adaptation while fixing original
RPS class-specific norms and biases.

Phase 3.10C does not authorize:

- ROP or ROP-v2;
- bounded/adaptive norm scaling;
- new representation learning;
- historical validation/test evaluation;
- extra seeds;
- additional datasets.

Stop after the Phase 3.10C mechanism decision.
