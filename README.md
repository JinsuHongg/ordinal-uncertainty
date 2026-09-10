# Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification

## Project status

Completed through **Phase 3.20A RetinaMNIST controlled imbalance-severity dose
response**. The
project proceeds as a bounded **mechanism / empirical analysis paper**. Across
the tested frozen RPS representations on RetinaMNIST, UTKFace, and Solar,
classifier direction adaptation improves rare upper-end localization. Phase
3.18A/3.18B reproduce that direction response on frozen CE representations in
RetinaMNIST and Solar. Thus direction responsiveness is directly observed on
both CE- and RPS-trained frozen representations in those two domains. Scale,
bias, global, and opposite-endpoint effects are dataset-dependent; no new
experiments are automatically authorized.

On the frozen RetinaMNIST RPS representation, a training-only 2×2 factorial
found balanced sampling—not CE versus RPS loss—to be the dominant signal
associated with beneficial direction rotation. This remains a bounded
single-dataset mechanism result, not a universal causal or method claim.

The controlled CE support-reduction study found consistently lower rare-class
probability and class-4-vs-class-3 margin at lower support, but a partial,
non-monotonic response in broader localization and feature geometry. It does
not establish a universal imbalance-severity dose response.

## Research question

The project studies **rare upper-extreme inward localization observed in
imbalanced ordinal settings** and the frozen-head mechanisms associated with
its recoverable component. It does not claim that class imbalance was
independently isolated as the causal source; class difficulty, representation
quality, dataset structure, and label ambiguity remain potential contributors.
The RetinaMNIST representation/head decomposition remains dataset-specific
evidence.

The confirmed cross-dataset finding is:

> Rare upper-extreme inward localization is consistently observed across the
> evaluated imbalanced ordinal settings: RetinaMNIST, UTKFace, and solar-flare
> classification. This does not imply universal RPS superiority.

Within RetinaMNIST and Solar, the controlled direction-only head response now
appears on both frozen CE and RPS representations. This remains a bounded
two-domain finding, not objective independence or a universal claim.

## Canonical setup

- RetinaMNIST, official train/validation/test splits
- Native 28×28 RGB inputs
- Unpretrained small-image ResNet18
- Primary seeds: 0–4

RetinaMNIST remains a **development benchmark**. Phase 3.8 was a matched CE/RPS solar confirmation study; RPS did not need to win, and it did not improve the solar risk-quality metrics. New-method development and all follow-up experiments require separate authorization.

## Repository structure

- `src/ordinal_uncertainty/`: models, metrics, evaluation, and data utilities
- `scripts/`: phase-specific training and analysis entry points
- `tests/`: focused unit and pipeline tests
- `docs/research/`: experiment state, decisions, and evidence records

## Setup

```bash
pip install -e .[dev]
```

## Tests

```bash
pytest -q
```

## Research documentation

- [Current research state](docs/research/current_state.md)
- [Experiment plan](docs/research/experiment_plan.md)
- [Decision log](docs/research/decision_log.md)
- [Phase 2 model-level study](docs/research/phase2_model_level_ordinal_uq.md)
- [Phase 3.3 representation audit](docs/research/phase3_3_representation_failure_audit.md)
- [Phase 3.4 head-intervention audit](docs/research/phase3_4_frozen_head_intervention_audit.md)
- [Phase 3.5 risk-conditioned representation design audit](docs/research/phase3_5_risk_conditioned_representation_design.md)
- [Phase 3.6 RG-ACR seed-0 falsification](docs/research/phase3_6_rg_acr_seed0.md)
- [Phase 3.7A UTKFace failure replication](docs/research/phase3_7a_utkface_failure_replication.md)
- [Phase 3.8 Solar rare-extreme shrinkage confirmation](docs/research/phase3_8_solar_rare_extreme_shrinkage_confirmation.md)
- [Phase 3.10A RetinaMNIST ROP objective falsification](docs/research/phase3_10a_retinamnist_rop_objective_falsification.md)
- [Phase 3.10B RetinaMNIST head-bias localization audit](docs/research/phase3_10b_head_bias_localization_audit.md)
- [Phase 3.10C RetinaMNIST direction-only head falsification](docs/research/phase3_10c_direction_only_head_falsification.md)
- [Phase 3.10D RetinaMNIST controlled-scale head falsification](docs/research/phase3_10d_controlled_scale_head_falsification.md)
- [Phase 3.10E RetinaMNIST controlled scale × ROP interaction](docs/research/phase3_10e_controlled_scale_rop_interaction.md)
- [Phase 3.11 RetinaMNIST frozen candidate validation](docs/research/phase3_11_frozen_candidate_validation.md)
- [Phase 3.12 evidence consolidation and candidate disposition](docs/research/phase3_12_evidence_consolidation_and_candidate_disposition.md)
- [Phase 3.13 UTKFace direction/scale mechanism confirmation](docs/research/phase3_13_utkface_direction_scale_mechanism_confirmation.md)
- [Phase 3.14 cross-dataset mechanism consolidation](docs/research/phase3_14_cross_dataset_mechanism_consolidation.md)
- [Phase 3.15 Solar direction/scale mechanism confirmation](docs/research/phase3_15_solar_direction_scale_mechanism_confirmation.md)
- [Phase 3.16 three-dataset mechanism consolidation](docs/research/phase3_16_three_dataset_mechanism_consolidation.md)
- [Phase 3.17 paper framing and novelty boundary](docs/research/phase3_17_paper_framing_and_novelty_boundary.md)
- [Phase 3.18A RetinaMNIST CE direction robustness](docs/research/phase3_18a_retinamnist_ce_direction_robustness.md)
- [Phase 3.18B Solar CE direction robustness](docs/research/phase3_18b_solar_ce_direction_robustness.md)
- [Phase 3.19 RetinaMNIST sampling vs objective disentanglement](docs/research/phase3_19_retinamnist_sampling_objective_direction_disentanglement.md)
- [Phase 3.20A RetinaMNIST imbalance-severity dose response](docs/research/phase3_20a_retinamnist_imbalance_severity_dose_response.md)
- [Paper story architecture](docs/research/paper_story_architecture.md)
