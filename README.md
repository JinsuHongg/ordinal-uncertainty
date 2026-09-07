# Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification

## Project status

Completed through **Phase 3.14 cross-dataset mechanism consolidation**. RetinaMNIST and UTKFace support direction adaptation, with added scale strengthening rare upper-end localization; opposite-endpoint, global, and bias effects are dataset-dependent. This is mechanism evidence, not a freeze-ready universal method. A frozen Solar A/B/C/D confirmation is scientifically justified but requires separate execution authorization.

## Research question

Existing ordinal and imbalance-aware methods can provide useful decision-risk information, but rare upper-extreme samples can still be systematically pulled toward central classes. The emerging cross-dataset focus is **rare upper-extreme inward localization bias under ordinal imbalance**; the RetinaMNIST representation/head decomposition remains dataset-specific evidence.

The confirmed cross-dataset finding is:

> Rare upper-extreme inward localization bias under ordinal imbalance appears on RetinaMNIST, UTKFace, and solar-flare classification; this does not imply universal RPS superiority.

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
