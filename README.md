# Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification

## Project status

Phase 3.2 candidate-method design is justified but has not started.

## Research question

Existing ordinal and imbalance-aware methods can provide useful decision-risk
information, but rare upper-extreme samples can still be systematically pulled
toward central classes. The current target is to reduce rare upper-extreme inward
shrinkage while preserving RPS-like ordinal risk alignment, severe-error
detection, selective prediction, and global predictive quality.

## Canonical setup

- RetinaMNIST, official train/validation/test splits
- Native 28×28 RGB inputs
- Unpretrained small-image ResNet18
- Primary seeds: 0–4

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
