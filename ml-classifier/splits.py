"""Reproducible stratified row splits and portable JSON split manifests."""


import json
import math
from pathlib import Path

import numpy as np
from dataset import Dataset, DatasetError
from schema import FEATURE_NAMES, DatasetRow
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "validation", "test")


def describe_source(path: str | Path) -> dict:
    """Identify a source using file metadata without reading its contents.

    This is a quick change check, not proof of byte-for-byte identity: edits that
    preserve the path, size, and modification time cannot be detected.
    """
    source = Path(path).resolve()
    stat = source.stat()
    return {"path": str(source), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_split_settings(seed: int, validation_size: float, test_size: float) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ValueError("Training seed must be between 0 and 2**32 - 1")
    for name, value in (("validation_size", validation_size), ("test_size", test_size)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be a finite fraction between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")


def _validate_classes(partitions: dict[str, tuple[DatasetRow, ...]]) -> None:
    for name, rows in partitions.items():
        if {row.attack for row in rows} != {False, True}:
            raise DatasetError(
                f"{name} split requires both classes; increase the sample size "
                "or adjust split fractions"
            )


def create_splits(
    dataset: Dataset, *, source: dict, seed: int = 0,
    validation_size: float = 0.1, test_size: float = 0.1,
) -> tuple[dict[str, tuple[DatasetRow, ...]], dict]:
    validate_split_settings(seed, validation_size, test_size)
    dataset.validate_for_training()
    indices = np.arange(len(dataset.rows))
    labels = np.asarray([row.attack for row in dataset.rows], dtype=bool)
    try:
        remaining, test = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=seed
        )
        train, validation = train_test_split(
            remaining, test_size=validation_size / (1 - test_size),
            stratify=labels[remaining], random_state=seed,
        )
    except ValueError as error:
        raise DatasetError(
            "Cannot create stratified train/validation/test splits; increase the "
            f"sample size or adjust split fractions: {error}"
        ) from error
    partitions = {
        name: tuple(dataset.rows[int(index)] for index in sorted(selected))
        for name, selected in zip(SPLIT_NAMES, (train, validation, test), strict=True)
    }
    _validate_classes(partitions)
    manifest = {
        "format_version": 2,
        "feature_names": list(FEATURE_NAMES),
        "source": source,
        "sampling": {"sample_size": dataset.summary.sample_size, "seed": dataset.summary.seed},
        "split_settings": {
            "seed": seed, "validation_size": validation_size, "test_size": test_size,
        },
        "assignments": {
            name: [row.row_number for row in rows] for name, rows in partitions.items()
        },
    }
    return partitions, manifest


def restore_splits(
    path: str | Path, dataset: Dataset, *, source: dict, seed: int = 0,
    validation_size: float = 0.1, test_size: float = 0.1,
) -> tuple[dict[str, tuple[DatasetRow, ...]], dict]:
    """Verify source/config and use stored row assignments, never re-split."""
    validate_split_settings(seed, validation_size, test_size)
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise DatasetError("Split manifest must be a JSON object")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != 2:
        raise DatasetError("Unsupported split manifest format_version")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise DatasetError("Split manifest feature schema does not match")
    saved_source = manifest.get("source")
    if not isinstance(saved_source, dict) or any(
        saved_source.get(key) != source[key] for key in ("path", "size_bytes", "mtime_ns")
    ):
        raise DatasetError("Split manifest source metadata does not match CSV path, size, or mtime")
    expected_sampling = {"sample_size": dataset.summary.sample_size, "seed": dataset.summary.seed}
    expected_settings = {
        "seed": seed, "validation_size": validation_size, "test_size": test_size,
    }
    if manifest.get("sampling") != expected_sampling:
        raise DatasetError("Split manifest sampling settings do not match")
    if manifest.get("split_settings") != expected_settings:
        raise DatasetError("Split manifest split settings do not match")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, dict) or set(assignments) != set(SPLIT_NAMES):
        raise DatasetError("Split manifest must assign train, validation, and test rows")
    by_record = {row.row_number: row for row in dataset.rows}
    seen = set()
    partitions = {}
    for name in SPLIT_NAMES:
        records = assignments[name]
        if not isinstance(records, list) or any(type(record) is not int for record in records):
            raise DatasetError(f"{name}: split record IDs must be integer lists")
        if len(records) != len(set(records)) or seen.intersection(records):
            raise DatasetError("Split assignments contain duplicate or overlapping record IDs")
        if not set(records).issubset(by_record):
            raise DatasetError("Split assignments contain record IDs outside the selected sample")
        seen.update(records)
        partitions[name] = tuple(by_record[record] for record in sorted(records))
    if seen != set(by_record):
        raise DatasetError("Split assignments do not cover the complete selected sample")
    _validate_classes(partitions)
    return partitions, manifest
