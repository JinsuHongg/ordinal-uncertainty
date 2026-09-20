# Solar CE natural-sampling direction-only follow-up: remaining-seeds protocol

**Status:** frozen before fitting Solar CE N heads for seeds 1, 3, and 4.
**Scope:** auxiliary follow-up only; it does not alter the original confirmatory
H1/H2a record.

## Purpose and frozen units

This protocol completes the previously defined Solar CE natural-sampling,
direction-only follow-up for the existing validated CE frozen-feature units
`1`, `3`, and `4`.  The four-unit descriptive summary will use those three
units plus the already complete coherent **replacement seed 2** unit at
`outputs/solar/mechanism_replication/replacement_seed2/ce/`.

The historical original CE seed 2 is not used here: it remains preserved as
historical provenance and invalid for this auxiliary follow-up because it did
not satisfy the frozen feature-identity gate.  The replacement unit is not
rerun or modified.

There will be no new backbone training, A retraining, C retraining,
hyperparameter search, metric selection, post-hoc seed substitution, change to
the N definition, or reroll based on outcome.  Exactly one N fit is authorized
for each of seeds 1, 3, and 4.

## Preconditions and stop gate

Before any fit, each seed must have a Phase-A feature manifest reporting
checkpoint integrity `READY` and archived-A output reproduction `PASS`; SHA256
validated train/validation/eval feature archives; 512-dimensional float32
features; disjoint split IDs; matching archived A/C readout IDs and labels;
available original A-head parameters; and an exactly recoverable C contract.
The expected rows are 45,047 train, 2,431 validation, and 28,006 eval, with
921 true upper-endpoint rows in eval.  Any failed or inconsistent condition
stops that seed; no feature regeneration, tolerance change, or substitute seed
is allowed.

## Frozen N intervention

For a seed's original frozen A head, \(z_k=w_k^\top h+b_k\), define
\(s_k^A=\lVert w_k^A\rVert_2\).  N uses the frozen feature representation and
the `DirectionOnlyLinear` parameterization:

\[
w_k^N=s_k^A\,v_k^N,\qquad \lVert v_k^N\rVert_2=1,\qquad b_k^N=b_k^A.
\]

It initializes the direction parameter from the original A classifier head,
holds the original A per-class norms and biases fixed, and updates only the
class-direction parameter.  The adaptation loss is cross entropy.

The already validated Solar N implementation is frozen without modification:

- implementation: `scripts/run_solar_natural_direction.py`;
- optimizer: AdamW over `head.direction` only;
- learning rate: `1e-3`; weight decay: `0.0`;
- batch size: `64`; epochs: `100`; scheduler: none;
- sampler generator: `torch.Generator().manual_seed(10000)`;
- initialization: original frozen A classifier head directions;
- numerical constraint tolerance: `1e-6`;
- checkpoint selection: none, fixed terminal epoch 100, exactly matching the
  archived Solar C contract;
- validation role: exported/count-audited only, not selected on, because C has
  no validation checkpoint selection.

N uses `natural_batch_indices`: one shuffled, without-replacement pass through
the empirical frozen training population per epoch.  It uses no class-balanced
resampling/batches, class weighting, oversampling, undersampling, logit
adjustment, or bias adjustment.  Relative to archived C, sampling is the only
changed operation; C uses replacement class-balanced batches under the same
fixed-norm/fixed-bias direction-only CE contract.

After every fit, fixed-norm preservation, fixed-bias preservation, optimizer
parameter isolation, and exact empirical-frequency draw counts are required.

## Outcomes and descriptive rule

The primary outcome is true-upper-endpoint exact-L1 MAE and
\(\Delta_{N-A}=\mathrm{MAE}_N-\mathrm{MAE}_A\).  Secondary frozen summaries
are C-versus-N endpoint MAE, global MAE, macro MAE, severe-error rate
(absolute exact-L1 error at least two), exact endpoint routing, predictive
endpoint mass, per-class endpoint-mass shift, and class-wise routing.

For the four-unit summary (`1`, replacement `2`, `3`, `4`):

- 4/4 negative \(\Delta_{N-A}\): **consistent natural-direction response**;
- exactly 3/4: **mixed/partial natural-direction response**;
- at most 2/4: **natural-direction response not consistent**.

This is an auxiliary descriptive gate, not a preregistered hypothesis test.
The summary also records whether N is weaker than C in each seed.  No new
decision rule may be introduced after results are observed.

## Reused code and outputs

Feature/archive validation and N fitting reuse
`scripts/run_solar_natural_direction.py`,
`src/ordinal_uncertainty/evaluation/direction_only.py`,
`src/ordinal_uncertainty/evaluation/oof.py`, and
`src/ordinal_uncertainty/metrics/decision.py`.  A/N/C metric and
redistribution summaries follow the established implementation in
`scripts/analyze_solar_rps_natural_direction.py`.

New N artifacts for seeds 1, 3, and 4 are written only to
`outputs/mechanism_replication/natural_direction/solar/ce/seed_<seed>/`.
Replacement seed-2 sources remain in their existing replacement namespace.
