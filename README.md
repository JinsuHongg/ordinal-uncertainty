# Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification

## Project status

Completed through **Phase 3.9 Solar mechanism audit**. Phase 3.9 found a **mixed but decomposable** solar failure: a centroid-collapsed X subset coexists with a larger X-like subset that the original head routes inward. Phase 3.8 provided **STRONG CONFIRMATION** of rare upper-extreme inward localization bias: both matched CE/RPS solar controls made zero exact X-class decisions under mode/L1/L2 and strongly inward-shrunk the upper endpoint. RetinaMNIST established a dataset-specific **dual-component rare-extreme failure** involving representation collapse and head-level inward bias; the predeclared RG-ACR seed-0 falsification was **NO-GO**. UTKFace provided a **PARTIAL REPLICATION**: rare upper-endpoint inward shrinkage and endpoint asymmetry persisted, but RPS did not reproduce its broad RetinaMNIST risk-quality advantage. No method is frozen.

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
