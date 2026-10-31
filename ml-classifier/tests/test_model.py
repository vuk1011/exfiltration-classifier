"""Model/artifact tests with tiny synthetic feature arrays and local temp files."""

import pickle
import platform
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import sklearn
from model import ModelArtifact, create_model, load_model, save_model
from schema import FEATURE_NAMES
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


class CreateModelTests(unittest.TestCase):
    def test_requested_parameters_and_unfitted_state(self):
        for seed in (0, 17):
            with self.subTest(seed=seed):
                estimator = create_model(seed)
                self.assertIsInstance(estimator, RandomForestClassifier)
                params = estimator.get_params()
                self.assertEqual(params["n_estimators"], 5)
                self.assertEqual(params["criterion"], "entropy")
                self.assertIsNone(params["class_weight"])
                self.assertEqual(params["random_state"], seed)
                self.assertIsNone(params["max_depth"])
                self.assertFalse(hasattr(estimator, "estimators_"))
        self.assertEqual(create_model().random_state, 0)

    def test_invalid_seed_types(self):
        for seed in (True, 1.5, "1", None):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                create_model(seed)  # pyright: ignore[reportArgumentType] -- intentionally invalid seed

    def test_seed_reproduces_predictions(self):
        features = np.random.default_rng(12).normal(size=(30, 9))
        labels = features[:, 0] > 0
        first = create_model(9).fit(features, labels)
        second = create_model(9).fit(features, labels)
        np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "forest.pkl"
        self.features = np.array([[float(i)] * 9 for i in range(12)])
        self.labels = np.array([False] * 6 + [True] * 6)
        self.estimator = create_model(4).fit(self.features, self.labels)
        self.metadata = {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }

    def write_artifact(self, artifact):
        # Deliberately bypass save validation to exercise corrupt local artifacts.
        with self.path.open("wb") as stream:
            pickle.dump(artifact, stream)

    def artifact(self, **changes):
        return replace(ModelArtifact(self.estimator, self.metadata.copy()), **changes)

    def test_dataclass_defaults(self):
        artifact = ModelArtifact(self.estimator, {})
        self.assertEqual(artifact.feature_names, FEATURE_NAMES)
        self.assertEqual(artifact.format_version, 1)

    def test_roundtrip_preserves_predictions_metadata_and_schema(self):
        metadata = {"seed": 4, "cohort": "synthetic", "sklearn_version": "caller-value"}
        original = metadata.copy()
        self.assertIsNone(save_model(self.path, self.estimator, metadata=metadata))
        artifact = load_model(str(self.path))
        self.assertIsInstance(artifact, ModelArtifact)
        self.assertEqual(artifact.feature_names, FEATURE_NAMES)
        self.assertEqual(artifact.format_version, 1)
        self.assertEqual(artifact.metadata, {**metadata, **self.metadata})
        self.assertEqual(metadata, original)
        np.testing.assert_array_equal(
            self.estimator.predict(self.features), artifact.estimator.predict(self.features)
        )
        np.testing.assert_array_equal(
            self.estimator.predict_proba(self.features), artifact.estimator.predict_proba(self.features)
        )
        self.assertEqual(artifact.estimator.get_params(), self.estimator.get_params())

    def test_exclusive_create_never_overwrites(self):
        self.path.write_bytes(b"existing artifact")
        with self.assertRaises(FileExistsError):
            save_model(self.path, self.estimator, metadata={})
        self.assertEqual(self.path.read_bytes(), b"existing artifact")

    def test_exclusive_create_does_not_follow_symlink(self):
        target = self.path.with_name("target.pkl")
        target.write_bytes(b"original")
        self.path.symlink_to(target)
        with self.assertRaises(FileExistsError):
            save_model(self.path, self.estimator, metadata={})
        self.assertEqual(target.read_bytes(), b"original")

    def test_save_rejects_unfitted_or_wrong_estimators_before_creating_file(self):
        for estimator in (create_model(), DummyClassifier().fit(self.features, self.labels)):
            with self.subTest(estimator=type(estimator).__name__):
                with self.assertRaises(ValueError):
                    save_model(self.path, estimator, metadata={})  # pyright: ignore[reportArgumentType] -- intentionally wrong estimator type
                self.assertFalse(self.path.exists())

    def test_save_requires_metadata_dict(self):
        with self.assertRaises(TypeError):
            save_model(self.path, self.estimator, metadata=None)  # pyright: ignore[reportArgumentType] -- intentionally invalid metadata
        self.assertFalse(self.path.exists())

    def test_load_rejects_wrong_container(self):
        self.write_artifact({"estimator": self.estimator})
        with self.assertRaisesRegex(ValueError, "ModelArtifact"):
            load_model(self.path)

    def test_load_rejects_tampered_feature_schema(self):
        for names in (FEATURE_NAMES[::-1], FEATURE_NAMES[:-1], list(FEATURE_NAMES)):
            with self.subTest(names=names):
                self.write_artifact(self.artifact(feature_names=names))
                with self.assertRaisesRegex(ValueError, "feature.*order"):
                    load_model(self.path)

    def test_load_rejects_unsupported_format_versions(self):
        for version in (0, 2, "1", True, 1.0):
            with self.subTest(version=version):
                self.write_artifact(self.artifact(format_version=version))
                with self.assertRaisesRegex(ValueError, "format_version"):
                    load_model(self.path)

    def test_load_rejects_missing_invalid_or_incompatible_versions(self):
        for key in self.metadata:
            for value in (None, "", 1):
                with self.subTest(key=key, value=value):
                    metadata = {**self.metadata, key: value}
                    self.write_artifact(self.artifact(metadata=metadata))
                    with self.assertRaisesRegex(ValueError, key):
                        load_model(self.path)
            metadata = self.metadata.copy()
            del metadata[key]
            self.write_artifact(self.artifact(metadata=metadata))
            with self.assertRaisesRegex(ValueError, key):
                load_model(self.path)
        self.write_artifact(self.artifact(metadata={**self.metadata, "sklearn_version": "0.0.0"}))
        with self.assertRaisesRegex(ValueError, "sklearn_version"):
            load_model(self.path)
        self.write_artifact(self.artifact(metadata=[]))
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_model(self.path)

    def test_python_and_numpy_versions_are_provenance_not_load_constraints(self):
        metadata = {**self.metadata, "python_version": "older-python", "numpy_version": "older-numpy"}
        self.write_artifact(self.artifact(metadata=metadata))
        self.assertEqual(load_model(self.path).metadata, metadata)

    def test_load_rejects_unfitted_wrong_type_and_empty_forest(self):
        empty = create_model().fit(self.features, self.labels)
        empty.estimators_ = []
        for estimator in (create_model(), empty, DummyClassifier()):
            with self.subTest(estimator=estimator):
                self.write_artifact(self.artifact(estimator=estimator))
                with self.assertRaises(ValueError):
                    load_model(self.path)

    def test_feature_count_and_named_feature_order_are_validated(self):
        wrong_count = create_model().fit(self.features[:, :-1], self.labels)
        wrong_order = create_model().fit(self.features, self.labels)
        wrong_order.feature_names_in_ = np.array(FEATURE_NAMES[::-1])
        for estimator in (wrong_count, wrong_order):
            with self.subTest(estimator=estimator):
                self.write_artifact(self.artifact(estimator=estimator))
                with self.assertRaisesRegex(ValueError, "feature"):
                    load_model(self.path)
                with self.assertRaisesRegex(ValueError, "feature"):
                    save_model(self.path.with_name("invalid.pkl"), estimator, metadata={})

    def test_only_fitted_single_output_binary_boolean_classes_are_accepted(self):
        labels = (
            self.labels.astype(int),
            self.labels.astype(str),
            np.zeros(12, dtype=bool),
            np.arange(12) % 3,
            np.column_stack([self.labels, ~self.labels]),
        )
        for values in labels:
            with self.subTest(labels=values.tolist()):
                estimator = create_model().fit(self.features, values)
                self.write_artifact(self.artifact(estimator=estimator))
                with self.assertRaisesRegex(ValueError, "binary bool"):
                    load_model(self.path)
                with self.assertRaisesRegex(ValueError, "binary bool"):
                    save_model(self.path.with_name("invalid.pkl"), estimator, metadata={})

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_model(self.path)


if __name__ == "__main__":
    unittest.main()
