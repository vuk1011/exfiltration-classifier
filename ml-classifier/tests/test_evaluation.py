"""Evaluation and leakage-audit tests using synthetic, in-memory DatasetRows."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from evaluation import audit_split_overlap, evaluate_model
from model import create_model
from schema import FEATURE_NAMES, DatasetRow, RequestMetadata


def make_row(number, attack=False, *, request=None, value=None, source="synthetic.csv"):
    metadata = None
    if request is not None:
        metadata = RequestMetadata("192.0.2.1", "example.test", "123", request)
    return DatasetRow(
        source=source,
        row_number=number,
        features=tuple(float(number if value is None else value) + i / 10 for i in range(9)),
        attack=attack,
        metadata=metadata,
    )


class FixedPredictions:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.calls = []

    def predict(self, features):
        self.calls.append(features.copy())
        return self.predictions


class EvaluationTests(unittest.TestCase):
    def evaluate(self, actual, predicted, **kwargs):
        rows = tuple(make_row(i, value) for i, value in enumerate(actual, 1))
        result = evaluate_model(FixedPredictions(predicted), rows, cohort="synthetic", **kwargs)
        json.dumps(result, allow_nan=False)
        return result

    def test_known_mixed_metrics_and_explicit_confusion_labels(self):
        result = self.evaluate(
            [False, False, False, True, True, True],
            [False, True, False, True, True, False],
        )
        self.assertEqual(result["cohort"], "synthetic")
        self.assertEqual(result["counts"], {"benign": 3, "attack": 3, "total": 6})
        self.assertEqual(result["confusion_matrix"], {"tp": 2, "fp": 1, "tn": 2, "fn": 1})
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_asymmetric_counts_and_metrics(self):
        result = self.evaluate(
            [True, True, True, False, False, False, False],
            [True, False, False, True, False, False, False],
        )
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "fp": 1, "tn": 3, "fn": 2})
        self.assertEqual(result["metrics"]["precision"], 0.5)
        self.assertAlmostEqual(result["metrics"]["recall"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["f1"], 0.4)
        self.assertEqual(result["metrics"]["false_positive_rate"], 0.25)

    def test_attack_only_metrics(self):
        for predicted, recall in (([True, True], 1.0), ([True, False], 0.5), ([False, False], 0.0)):
            with self.subTest(predicted=predicted):
                result = self.evaluate([True, True], predicted)
                self.assertEqual(result["counts"], {"benign": 0, "attack": 2, "total": 2})
                self.assertEqual(result["confusion_matrix"]["fp"], 0)
                self.assertEqual(result["confusion_matrix"]["tn"], 0)
                self.assertEqual(result["metrics"], {
                    "precision": None, "recall": recall, "f1": None,
                    "false_positive_rate": None, "accuracy": recall,
                })

    def test_benign_only_metrics(self):
        for predicted, fp in (([False, False], 0), ([False, True], 1), ([True, True], 2)):
            with self.subTest(predicted=predicted):
                result = self.evaluate([False, False], predicted)
                self.assertEqual(result["counts"], {"benign": 2, "attack": 0, "total": 2})
                self.assertEqual(result["confusion_matrix"], {"tp": 0, "fp": fp, "tn": 2 - fp, "fn": 0})
                self.assertEqual(result["metrics"], {
                    "precision": 0.0 if fp else None,
                    "recall": None,
                    "f1": 0.0 if fp else None,
                    "false_positive_rate": fp / 2,
                    "accuracy": (2 - fp) / 2,
                })

    def test_no_predicted_attacks_and_no_true_positives(self):
        result = self.evaluate([False, True], [False, False])
        self.assertIsNone(result["metrics"]["precision"])
        self.assertEqual(result["metrics"]["recall"], 0.0)
        self.assertEqual(result["metrics"]["f1"], 0.0)
        result = self.evaluate([False, True], [True, False])
        self.assertEqual(result["metrics"]["precision"], 0.0)
        self.assertEqual(result["metrics"]["f1"], 0.0)

    def test_perfect_mixed_and_single_row_cohorts(self):
        result = self.evaluate([False, True], [False, True])
        self.assertEqual(result["metrics"], {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "false_positive_rate": 0.0, "accuracy": 1.0,
        })
        self.assertEqual(result["errors"], [])
        for label in (False, True):
            result = self.evaluate([label], [label])
            self.assertEqual(sum(result["confusion_matrix"].values()), 1)

    def test_prediction_uses_only_ordered_precomputed_features_once_and_times_predict(self):
        rows = [make_row(1, False, request="raw-is-not-an-input"), make_row(2, True)]
        estimator = FixedPredictions([False, True])
        with patch("evaluation.perf_counter", side_effect=[10.0, 10.25]):
            result = evaluate_model(estimator, rows, cohort="timed")
        self.assertEqual(len(estimator.calls), 1)
        np.testing.assert_array_equal(estimator.calls[0], [row.features for row in rows])
        self.assertEqual(estimator.calls[0].shape, (2, 9))
        self.assertEqual(result["inference_seconds"], 0.25)
        self.assertEqual(result["inference_seconds_per_request"], 0.125)

    def test_errors_bounded_shared_limit_in_source_order_with_optional_metadata(self):
        rows = [
            make_row(4, False),
            make_row(7, True, request="raw.request"),
            make_row(9, False),
            make_row(15, True),
        ]
        result = evaluate_model(
            FixedPredictions([False, False, True, False]), rows, cohort="errors", error_limit=2
        )
        errors = result["errors"]
        self.assertEqual([error["row_number"] for error in errors], [7, 9])
        self.assertEqual(errors[0]["source"], "synthetic.csv")
        self.assertEqual(list(errors[0]["features"]), list(FEATURE_NAMES))
        self.assertEqual(list(errors[0]["features"].values()), list(rows[1].features))
        self.assertIs(errors[0]["actual"], True)
        self.assertIs(errors[0]["predicted"], False)
        self.assertIs(errors[1]["actual"], False)
        self.assertIs(errors[1]["predicted"], True)
        self.assertEqual(errors[0]["metadata"], {
            "user_ip": "192.0.2.1", "domain": "example.test", "timestamp": "123", "request": "raw.request",
        })
        self.assertNotIn("metadata", errors[1])
        self.assertEqual(result["confusion_matrix"], {"tp": 0, "fp": 1, "tn": 1, "fn": 2})
        json.dumps(result, allow_nan=False)

    def test_error_limit_zero_default_and_larger_than_error_count(self):
        self.assertEqual(self.evaluate([True], [False], error_limit=0)["errors"], [])
        self.assertEqual(len(self.evaluate([True] * 15, [False] * 15)["errors"]), 10)
        self.assertEqual(len(self.evaluate([True], [False], error_limit=20)["errors"]), 1)

    def test_empty_rows_and_invalid_error_limits_fail_before_prediction(self):
        estimator = FixedPredictions([])
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_model(estimator, [], cohort="empty")
        for limit in (-1, True, 1.5, "1", None):
            with self.subTest(limit=limit), self.assertRaises((TypeError, ValueError)):
                evaluate_model(estimator, [make_row(1)], cohort="bad", error_limit=limit)  # pyright: ignore[reportArgumentType] -- intentionally invalid limit
        self.assertEqual(estimator.calls, [])

    def test_invalid_features_and_labels(self):
        for row in (
            replace(make_row(1), features=(1.0,) * 8),
            replace(make_row(1), features=(float("nan"),) * 9),
            replace(make_row(1), features=(float("inf"),) * 9),
            replace(make_row(1), attack="False"),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate_model(FixedPredictions([False]), [row], cohort="invalid")

    def test_invalid_prediction_shape_and_labels(self):
        for predictions in ([], [True, False], [[True]], [2], [float("nan")], ["True"]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_model(FixedPredictions(predictions), [make_row(1)], cohort="invalid")

    def test_real_forest_integration_with_bool_predictions(self):
        rows = [make_row(i, i >= 6) for i in range(12)]
        estimator = create_model(3).fit([row.features for row in rows], [row.attack for row in rows])
        result = evaluate_model(estimator, rows, cohort="forest")
        self.assertEqual(result["counts"]["total"], 12)
        self.assertGreaterEqual(result["inference_seconds"], 0)
        json.dumps(result, allow_nan=False)


class SplitOverlapTests(unittest.TestCase):
    def test_duplicates_and_overlap_are_separate_exact_non_mutating_audits(self):
        train = [
            make_row(1, request="same.request", value=1),
            make_row(2, True, request="same.request", value=1),
            make_row(3, request="train.only", value=2),
            make_row(4, value=3),
        ]
        test = [
            make_row(1, request="same.request", value=2, source="test.csv"),
            make_row(2, request="test.only", value=1, source="test.csv"),
            make_row(3, request="test.only", value=1, source="test.csv"),
            make_row(4, True, value=4, source="test.csv"),
        ]
        splits = {"train": train, "test": test, "empty": []}
        original = {key: tuple(rows) for key, rows in splits.items()}
        result = audit_split_overlap(splits)
        self.assertIn("not deduplication", result["note"])
        self.assertEqual({key: tuple(rows) for key, rows in splits.items()}, original)
        for name in ("train", "test"):
            report = result["per_split"][name]
            self.assertEqual(report["total_rows"], 4)
            self.assertEqual(report["rows_without_metadata"], 1)
            self.assertEqual(report["raw_requests"], {
                "available_rows": 3, "unique_values": 2, "duplicate_rows": 1, "duplicate_groups": 1,
            })
            self.assertEqual(report["feature_vectors"], {
                "available_rows": 4, "unique_values": 3, "duplicate_rows": 1, "duplicate_groups": 1,
            })
        self.assertEqual(len(result["cross_split"]), 3)
        overlap = result["cross_split"][0]
        self.assertEqual((overlap["split_a"], overlap["split_b"]), ("train", "test"))
        self.assertEqual(overlap["raw_requests"], {
            "shared_unique_values": 1, "rows_in_split_a": 2, "rows_in_split_b": 1,
        })
        self.assertEqual(overlap["feature_vectors"], {
            "shared_unique_values": 2, "rows_in_split_a": 3, "rows_in_split_b": 3,
        })
        for overlap in result["cross_split"][1:]:
            for key in ("raw_requests", "feature_vectors"):
                self.assertEqual(overlap[key], {
                    "shared_unique_values": 0, "rows_in_split_a": 0, "rows_in_split_b": 0,
                })
        json.dumps(result, allow_nan=False)

    def test_requests_are_verbatim_and_missing_metadata_is_not_a_shared_request(self):
        result = audit_split_overlap({
            "a": [make_row(1, request="A.test"), make_row(2, request=""), make_row(3)],
            "b": [make_row(1, request="a.test"), make_row(2, request=" "), make_row(3)],
        })
        overlap = result["cross_split"][0]
        self.assertEqual(overlap["raw_requests"]["shared_unique_values"], 0)
        self.assertEqual(overlap["feature_vectors"]["shared_unique_values"], 3)

    def test_duplicate_rows_counts_extra_occurrences_not_all_group_members(self):
        result = audit_split_overlap({"only": [make_row(i, request="x", value=1) for i in range(5)]})
        self.assertEqual(result["cross_split"], [])
        for key in ("raw_requests", "feature_vectors"):
            self.assertEqual(result["per_split"]["only"][key]["duplicate_rows"], 4)
            self.assertEqual(result["per_split"]["only"][key]["duplicate_groups"], 1)

    def test_empty_splits_and_missing_metadata(self):
        self.assertEqual(audit_split_overlap({})["per_split"], {})
        self.assertEqual(audit_split_overlap({})["cross_split"], [])
        result = audit_split_overlap({"empty": [], "no_metadata": [make_row(1), make_row(2)]})
        self.assertEqual(result["per_split"]["empty"]["total_rows"], 0)
        self.assertEqual(result["per_split"]["no_metadata"]["rows_without_metadata"], 2)
        self.assertEqual(result["per_split"]["no_metadata"]["raw_requests"]["available_rows"], 0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
