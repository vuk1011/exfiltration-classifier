"""Strict streaming CSV ingestion and reproducible bounded-memory sampling."""

import csv
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from schema import (
    CSV_COLUMNS,
    FEATURE_NAMES,
    FEATURE_START,
    FEATURE_STOP,
    LABEL_INDEX,
    ClassCounts,
    DatasetRow,
    DatasetSummary,
    RequestMetadata,
)


class DatasetError(ValueError):
    """A dataset does not satisfy the input or training contract."""


def iter_rows(
    path: str | Path, *, include_metadata: bool = False
) -> Iterator[DatasetRow]:
    """Yield validated records without retaining the source file in memory.

    Files must be headerless and have all 22 published columns. Only the label
    and nine single-request features are parsed; aggregate values are ignored.
    Row numbers identify CSV records (not physical lines in quoted fields).
    """
    source = Path(path).resolve()
    row_number = 0
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            for row_number, fields in enumerate(reader, start=1):
                location = f"{source}: record {row_number}"
                if len(fields) != len(CSV_COLUMNS):
                    raise DatasetError(
                        f"{location}: expected {len(CSV_COLUMNS)} columns in a "
                        f"headerless CSV, got {len(fields)}"
                    )
                label = fields[LABEL_INDEX]
                if label not in ("True", "False"):
                    raise DatasetError(
                        f"{location}: attack must be 'True' or 'False', got {label!r}"
                    )

                features = []
                for name, value in zip(
                    FEATURE_NAMES, fields[FEATURE_START:FEATURE_STOP], strict=True
                ):
                    try:
                        number = float(value)
                    except ValueError as error:
                        raise DatasetError(
                            f"{location}: {name} must be numeric, got {value!r}"
                        ) from error
                    if not math.isfinite(number):
                        raise DatasetError(
                            f"{location}: {name} must be finite, got {value!r}"
                        )
                    features.append(number)

                metadata = None
                if include_metadata:
                    metadata = RequestMetadata(
                        user_ip=fields[0], domain=fields[1],
                        timestamp=fields[2], request=fields[4],
                    )
                yield DatasetRow(
                    source=str(source), row_number=row_number,
                    features=tuple(features), attack=label == "True",
                    metadata=metadata,
                )
        except csv.Error as error:
            raise DatasetError(
                f"{source}: CSV error near physical line {reader.line_num}: {error}"
            ) from error
        except UnicodeDecodeError as error:
            raise DatasetError(f"{source}: CSV must be UTF-8: {error}") from error

    if row_number == 0:
        raise DatasetError(f"{source}: dataset is empty")


@dataclass(frozen=True)
class Dataset:
    rows: tuple[DatasetRow, ...]
    summary: DatasetSummary

    def validate_for_training(self) -> None:
        """Single-class challenge sets are valid inputs, but not training sets."""
        counts = self.summary.selected_counts
        if not counts.benign or not counts.attack:
            raise DatasetError(
                "Training requires both benign and attack examples in the selected "
                "rows; increase the sample size or provide a mixed dataset"
            )


def load_dataset(
    path: str | Path,
    *,
    sample_size: int | None = None,
    seed: int = 0,
    include_metadata: bool = False,
) -> Dataset:
    """Read a dataset, optionally retaining a uniform reservoir of records.

    Sampling scans and validates the entire file, including unselected records.
    It does not balance classes or exclude domains. Memory is O(sample_size);
    passing None retains ALL rows and should only be used when memory permits.
    A seed reproduces selected record IDs for an unchanged file in the same order.
    """
    if sample_size is not None:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise TypeError("sample_size must be a positive integer or None")
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer or None")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = random.Random(seed)
    selected: list[DatasetRow] = []
    source_attack = 0
    source_total = 0
    for source_total, row in enumerate(
        iter_rows(path, include_metadata=include_metadata), start=1
    ):
        source_attack += row.attack
        if sample_size is None or len(selected) < sample_size:
            selected.append(row)
        else:
            # Each source record has the same chance of surviving in the reservoir.
            index = rng.randrange(source_total)
            if index < sample_size:
                selected[index] = row

    selected.sort(key=lambda row: row.row_number)
    selected_attack = sum(row.attack for row in selected)
    summary = DatasetSummary(
        source=selected[0].source,
        feature_names=FEATURE_NAMES,
        source_counts=ClassCounts(source_total - source_attack, source_attack),
        selected_counts=ClassCounts(len(selected) - selected_attack, selected_attack),
        sample_size=sample_size,
        seed=seed,
        include_metadata=include_metadata,
        sampling_method="all" if sample_size is None else "reservoir",
    )
    return Dataset(rows=tuple(selected), summary=summary)
