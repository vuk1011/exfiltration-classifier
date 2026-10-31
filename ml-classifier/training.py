"""Baseline training and independent CSV evaluation workflows."""

import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn

if __package__:
    from .dataset import Dataset, DatasetError, load_dataset
    from .evaluation import audit_split_overlap, evaluate_model
    from .model import create_model, load_model, save_model
    from .schema import FEATURE_NAMES
    from .splits import (
        create_splits,
        describe_source,
        restore_splits,
        validate_split_settings,
    )
else:
    from dataset import Dataset, DatasetError, load_dataset
    from evaluation import audit_split_overlap, evaluate_model
    from model import create_model, load_model, save_model
    from schema import FEATURE_NAMES
    from splits import (
        create_splits,
        describe_source,
        restore_splits,
        validate_split_settings,
    )

BASELINE_LIMITATION = (
    "Stratified random row split, not a session- or domain-held-out evaluation. "
    "Duplicate overlap is audited, not removed. Client/domain pairs are not true "
    "session identifiers. Results do not establish unseen-tool generalization."
)


def write_report(path: str | Path, report: dict) -> None:
    """Write JSON exclusively. The parent directory must already exist."""
    # Serialize before opening so invalid reports do not leave a partial file.
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def _load_source(
    path: str | Path, *, sample_size: int | None, seed: int,
) -> tuple[Dataset, dict]:
    source = Path(path).resolve()
    before = source.stat()
    source_metadata = describe_source(source)
    dataset = load_dataset(source, sample_size=sample_size, seed=seed, include_metadata=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise DatasetError(f"{source}: source changed while loading; retry with a stable CSV")
    return dataset, source_metadata


def _summary(dataset: Dataset) -> dict:
    summary = asdict(dataset.summary)
    summary["source_counts"]["total"] = dataset.summary.source_counts.total
    summary["selected_counts"]["total"] = dataset.summary.selected_counts.total
    summary["sampled_out_rows"] = dataset.summary.sampled_out_rows
    return summary


def _validate_error_limit(error_limit: int) -> None:
    if isinstance(error_limit, bool) or not isinstance(error_limit, int):
        raise TypeError("error_limit must be a nonnegative integer")
    if error_limit < 0:
        raise ValueError("error_limit must be a nonnegative integer")


def train_baseline(
    path: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int | None = 10_000,
    seed: int = 0,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    reuse_splits: str | Path | None = None,
    error_limit: int = 10,
) -> dict:
    """Fit only the training partition and report validation/test separately.

    No model selection, balancing, domain exclusion, scaling, or modified-attack
    enrichment is performed. Reused assignments must match source and settings.
    """
    validate_split_settings(seed, validation_size, test_size)
    _validate_error_limit(error_limit)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}; choose a new run directory")
    dataset, source = _load_source(path, sample_size=sample_size, seed=seed)
    split_options = {
        "source": source, "seed": seed,
        "validation_size": validation_size, "test_size": test_size,
    }
    if reuse_splits is None:
        partitions, manifest = create_splits(dataset, **split_options)
    else:
        partitions, manifest = restore_splits(reuse_splits, dataset, **split_options)
    estimator = create_model(seed)
    train_rows = partitions["train"]
    features = np.asarray([row.features for row in train_rows], dtype=float)
    labels = np.asarray([row.attack for row in train_rows], dtype=bool)
    start = perf_counter()
    estimator.fit(features, labels)
    training_seconds = perf_counter() - start
    evaluations = {
        name: evaluate_model(estimator, rows, cohort=name, error_limit=error_limit)
        for name, rows in partitions.items()
    }
    report = {
        "format_version": 1,
        "kind": "baseline_training",
        "input_contract": "published_precomputed_single_request_v1",
        "feature_names": list(FEATURE_NAMES),
        "source": source,
        "dataset": _summary(dataset),
        "split_settings": manifest["split_settings"],
        "estimator": {"type": "RandomForestClassifier", "parameters": estimator.get_params()},
        "environment": {
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        },
        "training_seconds": training_seconds,
        "evaluations": evaluations,
        "overlap_audit": audit_split_overlap(partitions),
        "global_feature_importance": dict(zip(
            FEATURE_NAMES, (float(value) for value in estimator.feature_importances_), strict=True
        )),
        "limitations": [BASELINE_LIMITATION, (
            "Features are loaded unchanged from the published CSV. No raw-query "
            "extractor or live preprocessing parity has been validated."
        )],
    }
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model.pkl"
    save_model(model_path, estimator, metadata={
        "input_contract": report["input_contract"],
        "source": source,
        "dataset": report["dataset"],
        "split_manifest": manifest,
        "limitations": report["limitations"],
    })
    report["model_size_bytes"] = model_path.stat().st_size
    write_report(output / "splits.json", manifest)
    write_report(output / "report.json", report)
    return report


def evaluate_file(
    path: str | Path,
    model_path: str | Path,
    *,
    sample_size: int | None = 10_000,
    seed: int = 0,
    cohort: str = "external",
    error_limit: int = 10,
) -> dict:
    """Evaluate a trusted saved model without fitting or changing it."""
    _validate_error_limit(error_limit)
    artifact = load_model(model_path)
    dataset, source = _load_source(path, sample_size=sample_size, seed=seed)
    same_source = source == artifact.metadata.get("source")
    assignments = artifact.metadata.get("split_manifest", {}).get("assignments", {})
    selected_ids = {row.row_number for row in dataset.rows}
    overlap = {
        name: len(selected_ids.intersection(records))
        for name, records in assignments.items()
    } if same_source else None
    return {
        "format_version": 1,
        "kind": "external_evaluation",
        "input_contract": "published_precomputed_single_request_v1",
        "feature_names": list(FEATURE_NAMES),
        "model": {"path": str(Path(model_path).resolve()), "metadata": artifact.metadata},
        "source": source,
        "dataset": _summary(dataset),
        "evaluation": evaluate_model(
            artifact.estimator, dataset.rows, cohort=cohort, error_limit=error_limit
        ),
        "same_source_as_training": same_source,
        "selected_record_overlap": overlap,
        "limitations": [(
            "Same-source evaluation may include training rows; inspect selected_record_overlap. "
            "Different-source evaluation does not by itself guarantee independent sessions or "
            "absence of duplicate queries. Source matching uses path, size, and mtime, "
            "not content verification. Attack-only data cannot measure precision or FPR."
        )],
    }
