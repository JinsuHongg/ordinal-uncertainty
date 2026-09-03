"""Frozen-manifest UTKFace data contract for the Phase 3.7A replication."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


AGE_THRESHOLDS = (20.0, 40.0, 60.0, 80.0)
NUM_CLASSES = 5


def parse_age(filename: str) -> float:
    """Parse a finite non-negative age from UTKFace's first filename field."""
    try:
        age = float(Path(filename).name.split("_", 1)[0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"unparseable UTKFace filename: {filename}") from error
    if not math.isfinite(age) or age < 0:
        raise ValueError(f"invalid UTKFace age: {filename}")
    return age


def age_bin(age: float) -> int:
    """Map age to [<20, 20--40, 40--60, 60--80, >=80]."""
    if not math.isfinite(age) or age < 0:
        raise ValueError("age must be finite and non-negative")
    for index, threshold in enumerate(AGE_THRESHOLDS):
        if age < threshold:
            return index
    return NUM_CLASSES - 1


def load_manifest(manifest_path: Path, data_root: Path) -> list[dict[str, Any]]:
    """Load and validate a frozen UTKFace manifest against its local corpus."""
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("UTKFace manifest is empty")
    ids = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if not sample_id.startswith("utkface:") or sample_id in ids:
            raise ValueError("manifest sample IDs must be unique utkface:<filename> values")
        ids.add(sample_id)
        filename = sample_id.removeprefix("utkface:")
        if not (data_root / filename).is_file():
            raise FileNotFoundError(f"manifest file missing from local corpus: {filename}")
        if age_bin(parse_age(filename)) != int(record["Y_ord"]):
            raise ValueError(f"manifest label does not match age bin: {filename}")
        if record["canonical_split"] not in {"train", "validation", "calibration", "test"}:
            raise ValueError("manifest has unknown split")
    return records


def records_for_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Return deterministically source-index-ordered records for one split."""
    selected = [record for record in records if record["canonical_split"] == split]
    if not selected:
        raise ValueError(f"no records for split {split}")
    return sorted(selected, key=lambda record: int(record["source_index"]))


def class_counts(records: list[dict[str, Any]]) -> list[int]:
    """Return all five class counts, preserving zero-count classes."""
    counts = Counter(int(record["Y_ord"]) for record in records)
    return [int(counts[index]) for index in range(NUM_CLASSES)]


def utkface_transform(train: bool) -> transforms.Compose:
    """Frozen historical 128px RGB/ImageNet-normalized transform."""
    stages: list[Any] = [transforms.Resize((128, 128))]
    if train:
        stages.append(transforms.RandomHorizontalFlip())
    stages.extend([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transforms.Compose(stages)


class UTKFaceDataset(Dataset[tuple[Any, int, str]]):
    """Manifest-backed RGB UTKFace image dataset with stable sample IDs."""

    def __init__(self, records: list[dict[str, Any]], data_root: Path, transform: Any) -> None:
        self.records = records
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        record = self.records[index]
        filename = str(record["sample_id"]).removeprefix("utkface:")
        with Image.open(self.data_root / filename) as image:
            value = image.convert("RGB")
        return self.transform(value), int(record["Y_ord"]), str(record["sample_id"])


def audit_corpus(data_root: Path) -> dict[str, Any]:
    """Inspect filename parsing and Pillow decodability without mutating data."""
    files = sorted(path for path in data_root.iterdir() if path.is_file())
    parsed: list[tuple[Path, float, int]] = []
    unparseable: list[str] = []
    decode_errors: list[str] = []
    formats: Counter[str] = Counter()
    for path in files:
        try:
            age = parse_age(path.name)
        except ValueError:
            unparseable.append(path.name)
            continue
        try:
            with Image.open(path) as image:
                formats[str(image.format)] += 1
                image.verify()
        except Exception:
            decode_errors.append(path.name)
            continue
        parsed.append((path, age, age_bin(age)))
    counts = Counter(label for _, _, label in parsed)
    ages = [age for _, age, _ in parsed]
    return {
        "data_root": str(data_root), "all_file_count": len(files), "valid_image_count": len(parsed),
        "unparseable_filename_count": len(unparseable), "unparseable_filenames": unparseable,
        "decode_error_count": len(decode_errors), "decode_error_filenames": decode_errors,
        "formats": dict(formats), "age_min": min(ages) if ages else None, "age_max": max(ages) if ages else None,
        "class_counts": [int(counts[index]) for index in range(NUM_CLASSES)],
        "filename_rule": "first underscore-delimited filename field is finite non-negative chronological age",
        "metadata_fields_present": "filename convention encodes age_gender_race_timestamp; gender/race are not model inputs",
        "image_variant": "provider-distributed UTKFace cropped/aligned filename corpus; no additional face alignment/crop",
    }
