# Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification

## Project status

Completed through **Phase 3.6**. The current diagnosis is a **dual-component rare-extreme failure** involving both representation collapse and head-level inward bias. The predeclared RG-ACR seed-0 falsification was **NO-GO**; the branch is stopped and no method is frozen.

## Research question

Existing ordinal and imbalance-aware methods can provide useful decision-risk information, but rare upper-extreme samples can still be systematically pulled toward central classes. Current evidence shows that this occurs at both the learned-representation and probabilistic-head levels.

The next research question is:

> Can ordinal decision risk guide representation learning so that rare, high-risk extreme samples become better localized without destroying probabilistic risk quality?

## Canonical setup

- RetinaMNIST, official train/validation/test splits
- Native 28×28 RGB inputs
- Unpretrained small-image ResNet18
- Primary seeds: 0–4

RetinaMNIST is now treated as a **development benchmark**; a promising next seed-0 method should be frozen before multi-seed and multi-dataset confirmation.

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
