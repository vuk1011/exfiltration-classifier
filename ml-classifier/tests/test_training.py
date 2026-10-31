"""End-to-end baseline training using only tiny, temporary 22-column CSVs."""

import csv
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dataset import DatasetError, load_dataset
from model import load_model
from schema import FEATURE_NAMES
from sklearn.ensemble import RandomForestClassifier
from training import evaluate_file, train_baseline, write_report


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "source.csv"
        self.output = self.directory / "run"
        self.write_csv(self.path)

    def write_csv(self, path, *, count=100, attack_only=False, duplicates=False):
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(
                ["192.0.2.1", "example.test", f"timestamp-{i}", str(attack_only or i > count // 2),
                 "raw, quoted\nrequest.example.test" if duplicates else f"raw, q{i}\nexample.test",
                 *[1 if duplicates else i + j / 10 for j in range(9)],
                 *["not a model input"] * 8]
                for i in range(1, count + 1)
            )
        return path

    def train(self, output=None, **options):
        return train_baseline(
            self.path, self.output if output is None else output,
            **{"sample_size": None, "seed": 0, **options},
        )

    def read_json(self, name):
        return json.loads((self.output / name).read_text(encoding="utf-8"))

    def assert_finite_json(self, report):
        # allow_nan=False rejects non-finite values anywhere in nested metadata too.
        return json.loads(json.dumps(report, allow_nan=False))

    def test_artifacts_report_schema_source_and_80_10_10_splits(self):
        report = self.train()
        self.assertEqual({path.name for path in self.output.iterdir()}, {
            "model.pkl", "splits.json", "report.json",
        })
        self.assertEqual(self.read_json("report.json"), self.assert_finite_json(report))
        manifest = self.read_json("splits.json")
        self.assertEqual(manifest["format_version"], 2)
        self.assert_finite_json(manifest)
        artifact = load_model(self.output / "model.pkl")
        self.assertEqual(artifact.feature_names, FEATURE_NAMES)
        self.assertEqual(report["feature_names"], list(artifact.feature_names))
        self.assertEqual(manifest["feature_names"], list(artifact.feature_names))
        self.assertEqual(artifact.metadata["split_manifest"], manifest)
        self.assertEqual(artifact.metadata["source"], report["source"])
        self.assertEqual(artifact.metadata["dataset"], report["dataset"])
        self.assertEqual(report["kind"], "baseline_training")
        self.assertEqual(report["format_version"], 1)
        self.assertEqual(report["input_contract"], "published_precomputed_single_request_v1")
        self.assertEqual(report["source"], {
            "path": str(self.path.resolve()), "size_bytes": self.path.stat().st_size,
            "mtime_ns": self.path.stat().st_mtime_ns,
        })
        self.assertEqual(manifest["source"], report["source"])
        self.assertEqual(report["split_settings"], manifest["split_settings"])
        self.assertEqual(report["dataset"]["selected_counts"], {"benign": 50, "attack": 50, "total": 100})
        self.assertEqual(report["dataset"]["sampled_out_rows"], 0)
        self.assertIsNone(report["dataset"]["sample_size"])
        self.assertTrue(report["dataset"]["include_metadata"])
        self.assertEqual(report["model_size_bytes"], (self.output / "model.pkl").stat().st_size)
        self.assertGreater(report["model_size_bytes"], 0)
        self.assertGreaterEqual(report["training_seconds"], 0)
        self.assertEqual(set(report["global_feature_importance"]), set(FEATURE_NAMES))
        self.assertAlmostEqual(sum(report["global_feature_importance"].values()), 1)
        records = []
        for name, size in (("train", 80), ("validation", 10), ("test", 10)):
            evaluation = report["evaluations"][name]
            self.assertEqual(evaluation["cohort"], name)
            self.assertEqual(evaluation["counts"], {"benign": size // 2, "attack": size // 2, "total": size})
            ids = manifest["assignments"][name]
            self.assertEqual(len(ids), size)
            self.assertEqual(ids, sorted(ids))
            records.extend(ids)
        self.assertEqual(sorted(records), list(range(1, 101)))

    def test_rerun_and_reuse_reproduce_ids_predictions_and_fitted_trees(self):
        self.train()
        manifest = self.read_json("splits.json")
        original = load_model(self.output / "model.pkl")
        features = np.asarray([row.features for row in load_dataset(self.path).rows])
        for name, options in (
            ("repeat", {}),
            ("reuse", {"reuse_splits": self.output / "splits.json"}),
        ):
            with self.subTest(run=name):
                output = self.directory / name
                self.train(output, **options)
                restored = load_model(output / "model.pkl")
                self.assertEqual(restored.metadata["split_manifest"], manifest)
                self.assertEqual(restored.estimator.get_params(), original.estimator.get_params())
                np.testing.assert_array_equal(
                    restored.estimator.predict_proba(features), original.estimator.predict_proba(features)
                )
                for before, after in zip(original.estimator.estimators_, restored.estimator.estimators_, strict=True):
                    np.testing.assert_array_equal(before.tree_.children_left, after.tree_.children_left)
                    np.testing.assert_array_equal(before.tree_.feature, after.tree_.feature)
                    np.testing.assert_array_equal(before.tree_.threshold, after.tree_.threshold)
                    np.testing.assert_array_equal(before.tree_.value, after.tree_.value)

    def test_fit_receives_only_training_rows_once(self):
        original_fit = RandomForestClassifier.fit
        calls = []

        def recording_fit(estimator, features, labels, *args, **kwargs):
            calls.append((features.copy(), labels.copy()))
            return original_fit(estimator, features, labels, *args, **kwargs)

        with patch.object(RandomForestClassifier, "fit", new=recording_fit):
            self.train()
        self.assertEqual(len(calls), 1)
        manifest = self.read_json("splits.json")
        by_id = {row.row_number: row for row in load_dataset(self.path).rows}
        train = [by_id[i] for i in manifest["assignments"]["train"]]
        np.testing.assert_array_equal(calls[0][0], [row.features for row in train])
        np.testing.assert_array_equal(calls[0][1], [row.attack for row in train])
        self.assertEqual(calls[0][0].shape, (80, 9))
        self.assertEqual(calls[0][1].dtype, np.dtype(bool))
        held_out = set(manifest["assignments"]["validation"] + manifest["assignments"]["test"])
        self.assertTrue(held_out.isdisjoint(int(value) for value in calls[0][0][:, 0]))

    def test_reuse_rejects_wrong_source_sampling_seed_and_fractions_before_fit(self):
        self.train()
        stat = self.path.stat()
        original = self.path.read_bytes()
        options = (
            {"sample_size": 80}, {"sample_size": 100}, {"seed": 1},
            {"validation_size": 0.2}, {"test_size": 0.2},
        )
        for i, changed in enumerate(options):
            output = self.directory / f"bad-config-{i}"
            with self.subTest(changed=changed), patch("training.create_model") as create:
                with self.assertRaisesRegex(DatasetError, "settings do not match"):
                    self.train(output, reuse_splits=self.output / "splits.json", **changed)
                create.assert_not_called()
                self.assertFalse(output.exists())
        self.path.write_bytes(original.replace(b"q1", b"x1", 1))
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.path.stat().st_size, stat.st_size)
        output = self.directory / "mutated"
        with patch("training.create_model") as create:
            with self.assertRaisesRegex(DatasetError, "source"):
                self.train(output, reuse_splits=self.output / "splits.json")
            create.assert_not_called()
        self.assertFalse(output.exists())

    def test_existing_run_directory_is_untouched_and_rejected_before_source_scan(self):
        self.train()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with patch("training._load_source") as load, patch("training.create_model") as create:
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self.train()
            load.assert_not_called()
            create.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)
        empty = self.directory / "empty-existing"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            self.train(empty)
        self.assertEqual(list(empty.iterdir()), [])

    def test_invalid_training_settings_fail_before_io_or_fit(self):
        cases = (
            ({"seed": True}, TypeError, "seed"),
            ({"seed": "0"}, TypeError, "seed"),
            ({"seed": -1}, ValueError, "seed"),
            ({"seed": 2**32}, ValueError, "seed"),
            ({"validation_size": False}, TypeError, "validation_size"),
            ({"validation_size": 0}, ValueError, "validation_size"),
            ({"test_size": float("nan")}, ValueError, "test_size"),
            ({"test_size": float("inf")}, ValueError, "test_size"),
            ({"validation_size": 0.6, "test_size": 0.4}, ValueError, "less than 1"),
            ({"error_limit": True}, TypeError, "error_limit"),
            ({"error_limit": 1.5}, TypeError, "error_limit"),
            ({"error_limit": None}, TypeError, "error_limit"),
            ({"error_limit": -1}, ValueError, "error_limit"),
        )
        for options, error, message in cases:
            with self.subTest(options=options), patch("training._load_source") as load:
                with self.assertRaisesRegex(error, message):
                    self.train(**options)
                load.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_sample_with_insufficient_minority_has_good_error_and_no_artifacts(self):
        # Two retained records cannot be stratified into three mixed partitions.
        with patch("training.create_model") as create:
            with self.assertRaisesRegex(DatasetError, "increase the sample size|increase the sample size or adjust"):
                self.train(sample_size=2)
            create.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_training_report_errors_preserve_raw_request_metadata_and_limit(self):
        # Invert the known labels, independently of how well the tiny forest fits.
        def inverted_predictions(features):
            return features[:, 0] <= 50

        with patch.object(RandomForestClassifier, "predict", side_effect=inverted_predictions):
            report = self.train(error_limit=3)
        dataset = load_dataset(self.path, include_metadata=True)
        by_id = {row.row_number: row for row in dataset.rows}
        manifest = self.read_json("splits.json")
        for name, evaluation in report["evaluations"].items():
            errors = evaluation["errors"]
            self.assertEqual(len(errors), 3)
            self.assertEqual([e["row_number"] for e in errors], manifest["assignments"][name][:3])
            for error in errors:
                row = by_id[error["row_number"]]
                assert row.metadata is not None
                self.assertEqual(error["metadata"], asdict(row.metadata))
                self.assertEqual(error["metadata"]["request"], f"raw, q{row.row_number}\nexample.test")
                self.assertEqual(error["source"], str(self.path.resolve()))
                self.assertEqual(error["features"], dict(zip(FEATURE_NAMES, row.features, strict=True)))
                self.assertEqual(error["actual"], row.attack)
                self.assertEqual(error["predicted"], not row.attack)
        self.assertEqual(self.read_json("report.json"), self.assert_finite_json(report))

    def test_duplicate_audit_is_in_report_and_does_not_remove_rows(self):
        self.write_csv(self.path, duplicates=True)
        report = self.train(error_limit=0)
        audit = report["overlap_audit"]
        self.assertIn("not deduplication", audit["note"])
        for name, size in (("train", 80), ("validation", 10), ("test", 10)):
            summary = audit["per_split"][name]
            self.assertEqual(summary["total_rows"], size)
            self.assertEqual(summary["rows_without_metadata"], 0)
            for key in ("raw_requests", "feature_vectors"):
                self.assertEqual(summary[key], {
                    "available_rows": size, "unique_values": 1,
                    "duplicate_rows": size - 1, "duplicate_groups": 1,
                })
            self.assertEqual(report["evaluations"][name]["counts"]["total"], size)
            self.assertEqual(report["evaluations"][name]["errors"], [])
        self.assertEqual(len(audit["cross_split"]), 3)
        for pair in audit["cross_split"]:
            for key in ("raw_requests", "feature_vectors"):
                self.assertEqual(pair[key]["shared_unique_values"], 1)
                self.assertEqual(pair[key]["rows_in_split_a"], audit["per_split"][pair["split_a"]]["total_rows"])
                self.assertEqual(pair[key]["rows_in_split_b"], audit["per_split"][pair["split_b"]]["total_rows"])
        self.assert_finite_json(report)

    def test_source_changed_during_parse_is_rejected_before_fit_or_output(self):
        def load_and_mutate(*args, **kwargs):
            dataset = load_dataset(*args, **kwargs)
            with self.path.open("ab") as stream:
                stream.write(b"changed after scan")
            return dataset

        with patch("training.load_dataset", side_effect=load_and_mutate), patch("training.create_model") as create:
            with self.assertRaisesRegex(DatasetError, "source changed while loading"):
                self.train()
            create.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_attack_only_external_evaluation_never_fits_and_has_defined_recall(self):
        self.train()
        path = self.write_csv(self.directory / "attack.csv", count=20, attack_only=True)
        model_path = self.output / "model.pkl"
        before = model_path.read_bytes()
        with patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("evaluation must not fit")):
            report = evaluate_file(path, model_path, sample_size=None, seed=0, cohort="attack-only")
        self.assertEqual(model_path.read_bytes(), before)
        self.assertEqual(report["kind"], "external_evaluation")
        self.assertEqual(report["evaluation"]["cohort"], "attack-only")
        self.assertEqual(report["evaluation"]["counts"], {"benign": 0, "attack": 20, "total": 20})
        metrics = report["evaluation"]["metrics"]
        self.assertEqual(metrics["recall"], report["evaluation"]["confusion_matrix"]["tp"] / 20)
        for metric in ("precision", "f1", "false_positive_rate"):
            self.assertIsNone(metrics[metric])
        self.assertFalse(report["same_source_as_training"])
        self.assertIsNone(report["selected_record_overlap"])
        artifact = load_model(model_path)
        self.assertEqual(report["feature_names"], list(artifact.feature_names))
        self.assertEqual(report["model"]["metadata"], artifact.metadata)
        self.assert_finite_json(report)

    def test_same_source_reports_exact_selected_record_overlap(self):
        self.train()
        manifest = self.read_json("splits.json")
        for sample_size in (None, 30):
            with self.subTest(sample_size=sample_size):
                report = evaluate_file(self.path, self.output / "model.pkl", sample_size=sample_size, seed=0)
                self.assertTrue(report["same_source_as_training"])
                selected = {row.row_number for row in load_dataset(self.path, sample_size=sample_size, seed=0).rows}
                self.assertEqual(report["selected_record_overlap"], {
                    name: len(selected.intersection(records))
                    for name, records in manifest["assignments"].items()
                })
                self.assertEqual(sum(report["selected_record_overlap"].values()), len(selected))
                self.assert_finite_json(report)

    def test_relocated_source_or_changed_metadata_is_not_same_source(self):
        self.train()
        stat = self.path.stat()
        relocated = self.directory / "relocated.csv"
        relocated.write_bytes(self.path.read_bytes())
        os.utime(relocated, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        model_path = self.output / "model.pkl"
        with patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("evaluation must not fit")):
            report = evaluate_file(relocated, model_path, sample_size=None, seed=0)
            self.assertFalse(report["same_source_as_training"])
            self.assertIsNone(report["selected_record_overlap"])
            self.assert_finite_json(report)
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            report = evaluate_file(self.path, model_path, sample_size=None, seed=0)
            self.assertFalse(report["same_source_as_training"])
            self.assertIsNone(report["selected_record_overlap"])
            self.assert_finite_json(report)

    def test_evaluation_invalid_error_limit_fails_before_loading_artifact(self):
        for value, error in ((True, TypeError), ("1", TypeError), (None, TypeError), (-1, ValueError)):
            with self.subTest(value=value), patch("training.load_model") as load:
                with self.assertRaisesRegex(error, "error_limit"):
                    evaluate_file(self.path, self.output / "model.pkl", error_limit=value)  # pyright: ignore[reportArgumentType]
                load.assert_not_called()


class WriteReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "report.json"

    def test_exclusive_write_preserves_existing_report(self):
        report = {"metrics": {"precision": None, "recall": 0.5}}
        self.assertIsNone(write_report(self.path, report))
        before = self.path.read_bytes()
        self.assertEqual(json.loads(before), report)
        with self.assertRaises(FileExistsError):
            write_report(self.path, {"replacement": True})
        self.assertEqual(self.path.read_bytes(), before)

    def test_nonfinite_report_is_rejected_without_partial_file(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                write_report(self.path, {"nested": [{"metric": value}]})
            self.assertFalse(self.path.exists())

    def test_missing_parent_is_not_created(self):
        path = self.path.parent / "missing" / "report.json"
        with self.assertRaises(FileNotFoundError):
            write_report(path, {"valid": True})
        self.assertFalse(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
