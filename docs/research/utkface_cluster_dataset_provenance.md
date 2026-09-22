# UTKFace Cluster Dataset Provenance

**Status:** **A — UTKFACE CORPUS DOWNLOADED; MANIFEST RESOLUTION PASS**

**Scope:** Dataset acquisition and frozen-manifest verification only. No model
training, CE/RPS backbone job, A/C/N fitting, split modification, loader smoke,
GPU smoke, manuscript edit, commit, or push was performed.

## Distribution and acquisition

The frozen manifest uses the flat, provider-distributed UTKFace
aligned-and-cropped JPEG filename corpus: each ID has the form
`utkface:<age>_<gender>_<race>_<timestamp>.jpg.chip.jpg`. The manifest contains
no directory prefix and its 23,708 IDs use the standard UTKFace filename
convention.

The official UTKFace project identifies this as its aligned-and-cropped
distribution and historically hosted `UTKFace.tar.gz` through Google Drive
(file ID `0BxYys69jI14kYVM3aVhKS1VhRUk`). On 2026-09-21 UTC, that official Drive
URL redirected to a content endpoint that returned HTTP 404. The archive was
therefore acquired from the archival mirror below, which exposes an archive of
the same name and aligned/cropped filename convention. Its compatibility was
accepted only after exact frozen-manifest resolution.

| Field | Value |
| --- | --- |
| Archive source URL | `https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/e80f21fb21fa3631380c3cb634b79be3ac2f4cba/UTKFace.tar.gz?download=true` |
| Distribution | UTKFace aligned-and-cropped JPEG filename corpus |
| Archive filename | `UTKFace.tar.gz` |
| Download timestamp | 2026-09-21 02:57:22 UTC |
| Archive size | 106,634,631 bytes |
| Archive SHA256 | `2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7` |
| Download path | `/scratch/users/jhong36/data/utkface/downloads/UTKFace.tar.gz` |
| Extraction path | `/scratch/users/jhong36/data/utkface/extracted/` |
| Canonical dataset root | `/scratch/users/jhong36/data/utkface/extracted/UTKFace` |

The archive was extracted unchanged into the empty extraction directory. It
preserves its original `UTKFace/` directory and every original filename.

## Corpus inventory

| Check | Result |
| --- | ---: |
| Extracted files | 23,708 |
| JPEG files | 23,708 |
| Parseable filenames | 23,708 |
| Filename parse failures | 0 |
| Pillow decode failures | 0 |
| Age range | 1--116 |
| Duplicate filenames | 0 |
| Nested data directories below canonical root | 0 |
| Extracted disk usage | 121 MiB |

The repository parser uses the first underscore-delimited filename field as a
finite, non-negative chronological age. Its frozen bins map 20, 40, 60, and 80
to classes 1, 2, 3, and 4 respectively.

## Frozen manifest resolution

| Check | Result |
| --- | ---: |
| Manifest path | `/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl` |
| Manifest SHA256 | `3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83` |
| Manifest rows / unique IDs | 23,708 / 23,708 |
| Resolved rows | 23,708 |
| Missing rows | 0 |
| Multiple resolutions | 0 |
| Extra corpus files | 0 |
| Manifest label/age-bin mismatches | 0 |
| Split overlap, including calibration | 0 |

Each manifest row contains `sample_id`, `Y_ord`, `Z`, `canonical_split`, and
`source_index`. `Z` agrees with the filename-parsed age and `Y_ord` agrees with
the frozen ordinal-bin reconstruction for every row.

## Frozen split verification

Membership checksums are SHA256 over lexicographically sorted sample IDs, each
terminated with a newline.

| Split | N | Class counts [0,1,2,3,4] | Class-4 support | Membership SHA256 |
| --- | ---: | --- | ---: | --- |
| Train | 14,224 | [2756, 7128, 2726, 1210, 404] | 404 | `726c8be7ccf695de35a7ddf10c240c42f7fd2808fde7266b474226cba8f16843` |
| Validation | 2,371 | [459, 1188, 455, 202, 67] | 67 | `f457d284418da1a12d36f7eecc5114f7e150d0b1dba4b27fe9c40148075783dd` |
| Calibration | 4,742 | [919, 2376, 909, 403, 135] | 135 | `4e23a7277f8ddd5526dea7c13e1ed8026d0362738fe0f391a32c1c7381dcb0c7` |
| Test | 2,371 | [459, 1189, 454, 202, 67] | 67 | `69500b04807216df9f8f30938df2d8ddc300c1061bcd7b51d0152bfdca7c47b5` |

The frozen manifest remains unmodified. The corpus is ready for the subsequent
cluster execution audit only; this provenance task does not authorize runner
changes, loader/GPU smoke checks, or any experiment.
