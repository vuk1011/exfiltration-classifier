"""Evaluation of precomputed features and non-mutating split leakage audits."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import combinations
from time import perf_counter

import numpy as np
from schema import FEATURE_NAMES, DatasetRow
from sklearn.metrics import confusion_matrix


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_model(
    estimator, rows: Sequence[DatasetRow], *, cohort: str, error_limit: int = 10
) -> dict:
    """Predict once, using only nine precomputed features in schema order.

    True is attack. Undefined metrics are None, not NaN; attack-only cohorts
    cannot measure precision, F1 or false-positive rate. Timing covers only
    predict(), with total and mean seconds reported at the top level. Errors
    retain input/source order, bounded by one shared FN/FP limit.
    """
    if isinstance(error_limit, bool) or not isinstance(error_limit, int):
        raise TypeError("error_limit must be a nonnegative integer")
    if error_limit < 0:
        raise ValueError("error_limit must be a nonnegative integer")
    if not rows:
        raise ValueError("Cannot evaluate empty rows")
    features = np.asarray([row.features for row in rows], dtype=float)
    if features.shape != (len(rows), len(FEATURE_NAMES)):
        raise ValueError("Rows must contain nine features in FEATURE_NAMES order")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite")
    if any(not isinstance(row.attack, (bool, np.bool_)) for row in rows):
        raise ValueError("Row attack labels must be bool")
    actual = np.asarray([row.attack for row in rows], dtype=bool)

    start = perf_counter()
    predicted = np.asarray(estimator.predict(features))
    inference_seconds = perf_counter() - start
    if predicted.shape != actual.shape or not np.isin(predicted, [False, True]).all():
        raise ValueError("Estimator must predict one binary label per row")
    predicted = predicted.astype(bool)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(
            actual, predicted, labels=[False, True]
        ).ravel()
    )
    attack = tp + fn
    benign = tn + fp
    total = attack + benign
    metrics = {
        "precision": _ratio(tp, tp + fp) if benign else None,
        "recall": _ratio(tp, attack),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn) if benign else None,
        "false_positive_rate": _ratio(fp, benign),
        "accuracy": (tp + tn) / total,
    }
    errors = []
    if error_limit:
        for row, prediction in zip(rows, predicted, strict=True):
            if row.attack == prediction:
                continue
            example = {
                "source": row.source,
                "row_number": row.row_number,
                "features": dict(zip(
                    FEATURE_NAMES, (float(value) for value in row.features), strict=True
                )),
                "actual": bool(row.attack),
                "predicted": bool(prediction),
            }
            if row.metadata is not None:
                example["metadata"] = asdict(row.metadata)
            errors.append(example)
            if len(errors) == error_limit:
                break
    return {
        "cohort": cohort,
        "counts": {"benign": benign, "attack": attack, "total": total},
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": metrics,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_request": inference_seconds / total,
        "errors": errors,
    }


def _duplicate_counts(values: Counter) -> dict:
    return {
        "available_rows": sum(values.values()),
        "unique_values": len(values),
        # Count extra occurrences, not every member of a duplicate group.
        "duplicate_rows": sum(count - 1 for count in values.values()),
        "duplicate_groups": sum(count > 1 for count in values.values()),
    }


def _overlap_counts(left: Counter, right: Counter) -> dict:
    shared = left.keys() & right.keys()
    return {
        "shared_unique_values": len(shared),
        "rows_in_split_a": sum(left[value] for value in shared),
        "rows_in_split_b": sum(right[value] for value in shared),
    }


def audit_split_overlap(splits: Mapping[str, Sequence[DatasetRow]]) -> dict:
    """Audit exact raw requests and feature tuples, without deduplicating.

    Requests are compared verbatim, not normalized; missing metadata is skipped
    for request matching only. Feature matches ignore labels and metadata.
    duplicate_rows counts occurrences beyond the first within a split. Pairwise
    overlap reports shared distinct keys and all matching rows on each side.
    Memory is proportional to the distinct requests and vectors in the splits.
    """
    counters = {}
    per_split = {}
    for name, rows in splits.items():
        requests = Counter(row.metadata.request for row in rows if row.metadata is not None)
        vectors = Counter(tuple(row.features) for row in rows)
        counters[name] = (requests, vectors)
        per_split[name] = {
            "total_rows": len(rows),
            "rows_without_metadata": len(rows) - sum(requests.values()),
            "raw_requests": _duplicate_counts(requests),
            "feature_vectors": _duplicate_counts(vectors),
        }
    cross_split = []
    for left, right in combinations(splits, 2):
        cross_split.append({
            "split_a": left,
            "split_b": right,
            "raw_requests": _overlap_counts(counters[left][0], counters[right][0]),
            "feature_vectors": _overlap_counts(counters[left][1], counters[right][1]),
        })
    return {
        "note": (
            "Audit only, not deduplication: no rows are removed or changed. "
            "Raw requests are exact strings from available metadata; feature "
            "vectors are exact nine-feature tuples, regardless of label. "
            "duplicate_rows counts extra occurrences beyond the first."
        ),
        "per_split": per_split,
        "cross_split": cross_split,
    }
