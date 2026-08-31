#!/usr/bin/env python3
"""Evaluation-only staged reproduction from a frozen SLACE checkpoint."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.evaluation.probability_pipeline import finalize_probability_evaluation
from ordinal_uncertainty.models.resnet import make_resnet18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)

    def marker(message: str) -> None:
        print(message, flush=True)

    marker("STAGE 1: training complete (frozen checkpoint reuse)")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    marker("STAGE 2: best checkpoint selected")
    model = make_resnet18(5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    marker("STAGE 3: checkpoint loaded")
    loaders, _ = retinamnist_loaders(Path("data/medmnist"), 128, 0, False, 28)
    marker("STAGE 4: test inference start")
    logits, labels = [], []
    with torch.no_grad():
        for images, target in loaders["test"]:
            logits.append(model(images).numpy())
            labels.append(target.reshape(-1).numpy())
    marker("STAGE 5: test inference complete")
    finalize_probability_evaluation(np.concatenate(labels), np.concatenate(logits), output, marker)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
