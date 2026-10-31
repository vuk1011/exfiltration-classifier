"""Runtime tests using tiny real forests; no dataset access or baseline writes."""

import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

import classifier as runtime
import numpy as np
from classifier import (
    FEATURE_SCHEMA,
    DNSClassifier,
    DNSQuery,
    FeatureExtractionError,
    FeatureExtractor,
    InvalidFeaturesError,
    Prediction,
)
from model import ModelArtifactError, load_model, save_model
from schema import FEATURE_NAMES
from sklearn.ensemble import RandomForestClassifier


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "forest.pkl"
        self.x = np.array([[float(i + j) for j in range(9)] for i in range(12)])
        self.y = np.array([False] * 6 + [True] * 6)
        self.estimator = RandomForestClassifier(n_estimators=3, random_state=7).fit(self.x, self.y)
        save_model(self.path, self.estimator, metadata={"input_contract": FEATURE_SCHEMA})
        self.features: dict[str, float] = {
            name: float(value) for name, value in zip(FEATURE_NAMES, self.x[3], strict=True)
        }
        self.classifier = DNSClassifier.load(self.path)

    def test_real_forest_parity_and_prediction_metadata(self):
        for row in self.x:
            features: dict[str, float] = {
                name: float(value) for name, value in zip(FEATURE_NAMES, row, strict=True)
            }
            prediction = self.classifier.classify(features)
            self.assertIsInstance(prediction, Prediction)
            self.assertEqual(prediction.score, float(self.estimator.predict_proba([row])[0, 1]))
            self.assertEqual(prediction.is_attack, bool(self.estimator.predict([row])[0]))
            self.assertEqual(prediction.threshold, 0.5)
            self.assertEqual(prediction.features, features)
            self.assertTrue(all(type(value) is float for value in prediction.features.values()))
            self.assertEqual(prediction.artifact_format_version, 1)
            self.assertEqual(prediction.feature_schema, FEATURE_SCHEMA)
            self.assertIsNone(prediction.query)
        self.assertEqual(self.classifier.feature_names, FEATURE_NAMES)
        self.assertEqual(self.classifier.feature_schema, FEATURE_SCHEMA)
        self.assertEqual(self.classifier.threshold, 0.5)
        for name in ("feature_names", "feature_schema", "threshold"):
            with self.assertRaises(AttributeError):
                setattr(self.classifier, name, None)

    def test_real_half_score_uses_strict_threshold_and_matches_predict(self):
        path = self.path.with_name("tie.pkl")
        estimator = RandomForestClassifier(n_estimators=1, bootstrap=False, random_state=0)
        estimator.fit(np.zeros((2, 9)), np.array([False, True]))
        save_model(path, estimator, metadata={"input_contract": FEATURE_SCHEMA})
        features: dict[str, float] = dict.fromkeys(FEATURE_NAMES, 0.0)
        self.assertFalse(estimator.predict([list(features.values())])[0])
        for threshold, attack in ((0.5, False), (0.49, True), (0.51, False), (0, True), (1, False)):
            with self.subTest(threshold=threshold):
                prediction = DNSClassifier.load(path, threshold=threshold).classify(features)
                self.assertEqual(prediction.score, 0.5)
                self.assertEqual(prediction.threshold, threshold)
                self.assertIs(prediction.is_attack, attack)

    def test_mapping_order_is_canonical_and_input_is_isolated(self):
        reversed_features = dict(reversed(list(self.features.items())))
        expected = self.classifier.classify(self.features)
        prediction = self.classifier.classify(reversed_features)
        self.assertEqual(tuple(prediction.features), FEATURE_NAMES)
        self.assertEqual(prediction.score, expected.score)
        reversed_features[FEATURE_NAMES[0]] = 999
        self.assertEqual(prediction.features, self.features)
        with self.assertRaises(TypeError):
            prediction.features[FEATURE_NAMES[0]] = 1  # pyright: ignore[reportIndexIssue] -- intentionally mutate immutable mapping
        with self.assertRaises(FrozenInstanceError):
            prediction.score = 1  # pyright: ignore[reportAttributeAccessIssue] -- intentionally mutate frozen result

    def test_prediction_constructor_also_copies_features(self):
        prediction = Prediction(0.2, 0.5, False, self.features, 1, FEATURE_SCHEMA)
        self.features.clear()
        self.assertEqual(tuple(prediction.features), FEATURE_NAMES)

    def test_to_dict_is_json_safe_and_independent_without_query(self):
        prediction = self.classifier.classify(self.features)
        payload = prediction.to_dict()
        self.assertEqual(payload, {
            "score": prediction.score,
            "threshold": prediction.threshold,
            "is_attack": prediction.is_attack,
            "features": self.features,
            "artifact_format_version": 1,
            "feature_schema": FEATURE_SCHEMA,
            "query": None,
        })
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        features = payload["features"]
        assert isinstance(features, dict)
        features[FEATURE_NAMES[0]] = 999
        self.assertEqual(prediction.features, self.features)
        self.assertNotEqual(prediction.to_dict()["features"], features)

    def test_to_dict_preserves_query_and_exports_typed_extractor(self):
        query = DNSQuery("MiXeD.Example.COM.", "tXt")
        extractor: FeatureExtractor = lambda query: self.features
        prediction = DNSClassifier.load(self.path, feature_extractor=extractor).classify(query)
        payload = prediction.to_dict()
        self.assertEqual(payload["query"], {"name": query.name, "record_type": query.record_type})
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        metadata = payload["query"]
        assert isinstance(metadata, dict)
        metadata["name"] = "changed"
        self.assertEqual(query.name, "MiXeD.Example.COM.")
        self.assertEqual(prediction.to_dict()["query"], {"name": query.name, "record_type": "tXt"})

    def test_numeric_numpy_and_real_values(self):
        from fractions import Fraction

        for number in (0, 1.2, np.int32(2), np.int64(3), np.float32(1.2), np.float64(1.2), Fraction(1, 3)):
            with self.subTest(number=number):
                features = {str(name): number for name in FEATURE_NAMES}
                prediction = self.classifier.classify(features)
                self.assertEqual(prediction.features[FEATURE_NAMES[0]], float(number))

    def test_invalid_numeric_values(self):
        for number in (True, False, np.bool_(True), "1", None, complex(1), [], {},
                       float("nan"), float("inf"), -float("inf"), np.float32("nan"),
                       1e300, -1e300, 10 ** 1000):
            with self.subTest(number=repr(number)):
                features = {**self.features, FEATURE_NAMES[0]: number}
                with self.assertRaisesRegex(InvalidFeaturesError, FEATURE_NAMES[0]):
                    self.classifier.classify(features)  # pyright: ignore[reportArgumentType] -- intentionally invalid feature values

    def test_float32_overflow_is_meaningful_without_sklearn_call(self):
        with patch.object(self.classifier._artifact.estimator, "predict_proba") as predict:
            with self.assertRaisesRegex(InvalidFeaturesError, "float32"):
                self.classifier.classify({**self.features, "len": 1e100})
            predict.assert_not_called()

    def test_missing_extra_and_malformed_inputs(self):
        missing = self.features.copy()
        del missing[FEATURE_NAMES[0]]
        renamed = {**missing, "wrong": 1}
        for value in (missing, renamed, {**self.features, "extra": 1}, {}, None,
                      "Example.COM", b"packet", list(self.features.values()), object()):
            with self.subTest(value=value), self.assertRaises(InvalidFeaturesError):
                self.classifier.classify(value)  # pyright: ignore[reportArgumentType] -- intentionally malformed input

    def test_schema_rejected_before_callback(self):
        callback = Mock(return_value=self.features)
        classifier = DNSClassifier.load(self.path, feature_extractor=callback)
        for schema in (None, "", "other", FEATURE_NAMES, 1):
            with self.subTest(schema=schema), self.assertRaisesRegex(InvalidFeaturesError, "feature_schema"):
                classifier.classify(DNSQuery("Example.COM"), feature_schema=schema)  # pyright: ignore[reportArgumentType] -- intentionally invalid schema
        callback.assert_not_called()

    def test_threshold_validation(self):
        for threshold in (True, np.bool_(False), "0.5", None, complex(0.5),
                          float("nan"), float("inf"), -0.01, 1.01, 10 ** 1000):
            with self.subTest(threshold=repr(threshold)), self.assertRaisesRegex(ValueError, "threshold"):
                DNSClassifier.load(self.path, threshold=threshold)  # pyright: ignore[reportArgumentType] -- intentionally invalid threshold
        for threshold in (np.float32(0.5), np.int64(1)):
            self.assertEqual(DNSClassifier.load(self.path, threshold=threshold).threshold, float(threshold))

    def test_missing_and_wrong_artifact_contract(self):
        artifact = load_model(self.path)
        for metadata in ({}, {"input_contract": None}, {"input_contract": "other"},
                         {"input_contract": FEATURE_NAMES}):
            with self.subTest(metadata=metadata):
                path = self.path.with_name("wrong.pkl")
                artifact.metadata.pop("input_contract", None)
                artifact.metadata.update(metadata)
                with path.open("wb") as stream:
                    pickle.dump(artifact, stream)
                with self.assertRaisesRegex(ModelArtifactError, "input_contract"):
                    DNSClassifier.load(path)

    def test_callback_gets_exact_query_once_and_not_mapping(self):
        query = DNSQuery("MiXeD.Sub.Example.COM.", "tXt")
        callback = Mock(return_value=self.features)
        classifier = DNSClassifier.load(self.path, feature_extractor=callback)
        prediction = classifier.classify(query)
        callback.assert_called_once_with(query)
        self.assertIs(callback.call_args.args[0], query)
        self.assertIs(prediction.query, query)
        self.assertEqual(query.name, "MiXeD.Sub.Example.COM.")
        self.assertEqual(query.record_type, "tXt")
        mapping_prediction = classifier.classify(self.features)
        callback.assert_called_once()
        self.assertEqual(prediction.features, mapping_prediction.features)
        self.assertEqual(prediction.score, mapping_prediction.score)
        self.assertIsNone(mapping_prediction.query)

    def test_query_without_extractor_never_returns_prediction(self):
        with self.assertRaisesRegex(FeatureExtractionError, "feature_extractor"):
            self.classifier.classify(DNSQuery("Example.COM"))

    def test_extractor_failure_has_explicit_error_and_cause(self):
        failure = RuntimeError("extractor bug")
        callback = Mock(side_effect=failure)
        classifier = DNSClassifier.load(self.path, feature_extractor=callback)
        with self.assertRaises(FeatureExtractionError) as caught:
            classifier.classify(DNSQuery("Example.COM"))
        self.assertIs(caught.exception.__cause__, failure)
        callback.assert_called_once()

    def test_extractor_invalid_output_uses_same_feature_validation(self):
        for output in (None, DNSQuery("other"), {}, "raw", {**self.features, "len": True}):
            with self.subTest(output=output):
                classifier = DNSClassifier.load(self.path, feature_extractor=Mock(return_value=output))
                with self.assertRaises(InvalidFeaturesError):
                    classifier.classify(DNSQuery("Example.COM"))
        with self.assertRaisesRegex(TypeError, "feature_extractor"):
            DNSClassifier.load(self.path, feature_extractor=42)  # pyright: ignore[reportArgumentType] -- intentionally non-callable extractor

    def test_query_validation_and_frozen_verbatim_metadata(self):
        for field in ("name", "record_type"):
            for value in (None, 1, b"A", "", " \t"):
                kwargs = {"name": "Example.COM", field: value}
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    DNSQuery(**kwargs)  # pyright: ignore[reportArgumentType] -- intentionally invalid query fields
        query = DNSQuery(" Example.COM. ")
        self.assertEqual(query.name, " Example.COM. ")
        self.assertEqual(query.record_type, "A")
        with self.assertRaises(FrozenInstanceError):
            query.name = "changed"  # pyright: ignore[reportAttributeAccessIssue] -- intentionally mutate frozen query

    def test_load_once_and_no_fit_or_runtime_training_imports(self):
        with patch.object(runtime, "load_model", wraps=load_model) as loader:
            with patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("inference fitted")):
                classifier = DNSClassifier.load(self.path)
                for _ in range(3):
                    classifier.classify(self.features)
            loader.assert_called_once_with(self.path)

    def test_bad_predict_proba_outputs_raise_artifact_error(self):
        invalid = (None, 0.5, [0.5, 0.5], [[0.5]], [[0.5, 0.5], [0.5, 0.5]],
                   [[float("nan"), 0.5]], [[0.5, float("inf")]], [[-0.1, 1.1]],
                   [["0.5", "0.5"]], [[True, False]], [[0.5j, 0.5]], [[0.5], [0.5, 0.5]])
        for output in invalid:
            with (
                self.subTest(output=output),
                patch.object(self.classifier._artifact.estimator, "predict_proba", return_value=output),
                self.assertRaisesRegex(ModelArtifactError, "predict_proba"),
            ):
                self.classifier.classify(self.features)

    def test_positive_class_index_and_invalid_classes(self):
        estimator = self.classifier._artifact.estimator
        with (
            patch.object(estimator, "classes_", np.array([True, False])),
            patch.object(estimator, "predict_proba", return_value=np.array([[0.8, 0.2]])),
        ):
            self.assertEqual(self.classifier.classify(self.features).score, 0.8)
        for classes in ([False], [False, False], [True, True], [0, 1], [[False, True]], [[False], [True, False]]):
            with (
                self.subTest(classes=classes),
                patch.object(estimator, "classes_", classes),
                self.assertRaisesRegex(ModelArtifactError, "classes"),
            ):
                self.classifier.classify(self.features)

    def test_unexpected_estimator_bug_is_not_swallowed(self):
        with (
            patch.object(self.classifier._artifact.estimator, "predict_proba", side_effect=RuntimeError("bug")),
            self.assertRaisesRegex(RuntimeError, "bug"),
        ):
            self.classifier.classify(self.features)


class PackageExportsTests(unittest.TestCase):
    def test_package_exports_and_relative_imports_without_training(self):
        source = Path(runtime.__file__).resolve().parent
        name = "_classifier_test_package"
        spec = importlib.util.spec_from_file_location(
            name, source / "__init__.py", submodule_search_locations=[str(source)],
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        before = set(sys.modules)
        sys.modules[name] = package
        try:
            spec.loader.exec_module(package)
            for public in ("DNSClassifier", "DNSQuery", "Prediction", "InvalidFeaturesError",
                           "FeatureExtractionError", "FeatureExtractor", "FEATURE_NAMES", "FEATURE_SCHEMA", "ModelArtifactError"):
                self.assertIn(public, package.__all__)
                self.assertTrue(hasattr(package, public))
            self.assertEqual(package.DNSClassifier.__module__, name + ".classifier")
            self.assertEqual(package.FEATURE_SCHEMA, FEATURE_SCHEMA)
            added = set(sys.modules) - before
            self.assertFalse(any(module.rsplit(".", 1)[-1] in {"dataset", "training"} for module in added))
        finally:
            for module in list(sys.modules):
                if module == name or module.startswith(name + "."):
                    del sys.modules[module]


if __name__ == "__main__":
    unittest.main()
