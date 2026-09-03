import json
from pathlib import Path

import pytest
from PIL import Image

from ordinal_uncertainty.data.utkface import (
    UTKFaceDataset,
    age_bin,
    audit_corpus,
    class_counts,
    load_manifest,
    parse_age,
    records_for_split,
    utkface_transform,
)


def _image(path: Path) -> None:
    Image.new("RGB", (16, 12), "red").save(path)


def test_age_parsing_and_frozen_bins():
    assert parse_age("80_1_2_any.jpg.chip.jpg") == 80.0
    assert [age_bin(value) for value in (19, 20, 39, 40, 59, 60, 79, 80)] == [0, 1, 1, 2, 2, 3, 3, 4]
    with pytest.raises(ValueError):
        parse_age("bad_name.jpg")


def test_manifest_loader_dataset_shape_counts_and_ids(tmp_path: Path):
    records = []
    for index, age in enumerate((10, 25, 45, 65, 85)):
        name = f"{age}_0_0_{index}.jpg"
        _image(tmp_path / name)
        records.append({"sample_id": f"utkface:{name}", "source_index": index, "canonical_split": "train" if index < 4 else "test", "Y_ord": age_bin(age)})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    loaded = load_manifest(manifest, tmp_path)
    assert class_counts(loaded) == [1, 1, 1, 1, 1]
    dataset = UTKFaceDataset(records_for_split(loaded, "test"), tmp_path, utkface_transform(False))
    image, label, sample_id = dataset[0]
    assert image.shape == (3, 128, 128)
    assert label == 4 and sample_id.startswith("utkface:")


def test_manifest_rejects_label_that_disagrees_with_frozen_age_bin(tmp_path: Path):
    name = "85_0_0_bad_label.jpg"
    _image(tmp_path / name)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"sample_id": f"utkface:{name}", "source_index": 0, "canonical_split": "test", "Y_ord": 3}) + "\n")
    with pytest.raises(ValueError, match="label does not match age bin"):
        load_manifest(manifest, tmp_path)


def test_audit_counts_unparseable_files(tmp_path: Path):
    _image(tmp_path / "90_0_0_valid.jpg")
    _image(tmp_path / "bad_name.jpg")
    audit = audit_corpus(tmp_path)
    assert audit["valid_image_count"] == 1
    assert audit["unparseable_filename_count"] == 1
