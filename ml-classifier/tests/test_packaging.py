"""Flat-source compatibility and optional offline wheel integration tests.

Import and wheel checks need no trained artifact. Baseline-specific checks skip
when the trusted local baseline is absent; no dataset scan or model fitting is
needed. Wheel building is skipped when local build tools are unavailable.
"""

import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
import tomllib  # pyright: ignore[reportMissingImports] -- stdlib in required Python >=3.12
import unittest
import zipfile
from pathlib import Path

from model import ModelArtifact, _ArtifactUnpickler

SOURCE = Path(__file__).resolve().parents[1]
BASELINE = SOURCE / "runs" / "baseline" / "model.pkl"
PACKAGE = "dns_exfiltration_classifier"


def run_python(test, script, *, cwd, arguments=()):
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, *map(str, arguments)],
        cwd=cwd, capture_output=True, text=True, timeout=120, check=False,
    )
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return result


# Run with only the temporary installation added, never the flat source root.
IMPORT_CHECK = '''
import importlib
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
source = Path(sys.argv[2]).resolve()
assert Path.cwd() != source
assert source not in [Path(item).resolve() for item in sys.path]
for name in ("schema", "model", "dataset", "evaluation", "splits", "training", "app"):
    assert importlib.util.find_spec(name) is None, name
    module = importlib.import_module("dns_exfiltration_classifier." + name)
    assert module.__package__ == "dns_exfiltration_classifier"
    assert name not in sys.modules, name
'''

ARTIFACT_CHECK = '''
import sys
sys.path.insert(0, sys.argv[1])
from dns_exfiltration_classifier.model import ModelArtifact, load_model, save_model
from dns_exfiltration_classifier.schema import FEATURE_NAMES
artifact = load_model(sys.argv[2])
assert type(artifact) is ModelArtifact
assert artifact.feature_names == FEATURE_NAMES
assert artifact.format_version == 1
save_model(sys.argv[3], artifact.estimator, metadata=artifact.metadata)
restored = load_model(sys.argv[3])
assert restored.metadata == artifact.metadata
import numpy as np
vector = np.asarray([[42., 2., 4., 10., 3.5, .3, .1, .2, 0.]])
np.testing.assert_array_equal(
    artifact.estimator.predict_proba(vector), restored.estimator.predict_proba(vector)
)
'''

RUNTIME_CHECK = '''
import sys
sys.path.insert(0, sys.argv[1])
from dns_exfiltration_classifier import DNSClassifier, DNSQuery, FeatureExtractionError
from dns_exfiltration_classifier.model import load_model
from dns_exfiltration_classifier.schema import FEATURE_NAMES
import numpy as np
artifact = load_model(sys.argv[2])
values = (42., 2., 4., 10., 3.5, .3, .1, .2, 0.)
features = dict(zip(FEATURE_NAMES, values, strict=True))
classifier = DNSClassifier.load(sys.argv[2], threshold=.5, feature_extractor=None)
expected = float(artifact.estimator.predict_proba(np.asarray([values]))[0, 1])
prediction = classifier.classify(features)
assert prediction.score == expected
assert prediction.is_attack == (expected > .5)
assert prediction.threshold == .5
assert prediction.artifact_format_version == artifact.format_version
assert prediction.feature_schema == "published_precomputed_single_request_v1"
assert dict(prediction.features) == features
# Test equality at the actual score too, without fitting another estimator.
assert not DNSClassifier.load(sys.argv[2], threshold=expected).classify(features).is_attack
assert "model" not in sys.modules
assert "schema" not in sys.modules
'''


QUERY_CHECK = RUNTIME_CHECK + '''
query = DNSQuery("MiXeD.Sub.Example.COM.", "TXT")
seen = []
def extract(value):
    assert value is query
    assert value.name == "MiXeD.Sub.Example.COM."
    assert value.record_type == "TXT"
    seen.append(value)
    return features.copy()
with_extractor = DNSClassifier.load(sys.argv[2], threshold=.5, feature_extractor=extract)
named = with_extractor.classify(features)
assert seen == []
extracted = with_extractor.classify(query)
assert seen == [query]
assert extracted.query is query
assert extracted.score == named.score == prediction.score
assert extracted.is_attack == named.is_attack
assert dict(extracted.features) == features
assert extracted.feature_schema == "published_precomputed_single_request_v1"
try:
    classifier.classify(query)
except FeatureExtractionError as error:
    assert "feature_extractor" in str(error)
else:
    raise AssertionError("DNSQuery without an extractor must fail explicitly")
'''

RUNTIME_IMPORT_CHECK = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
source = Path(sys.argv[2]).resolve()
assert Path.cwd() != source
assert source not in [Path(item).resolve() for item in sys.path]
from dns_exfiltration_classifier import DNSClassifier, DNSQuery, FeatureExtractionError
assert DNSClassifier.__module__ == "dns_exfiltration_classifier.classifier"
assert DNSQuery("example.test", "TXT").record_type == "TXT"
assert issubclass(FeatureExtractionError, ValueError)
for name in ("dataset", "training", "evaluation"):
    assert "dns_exfiltration_classifier." + name not in sys.modules, name
for name in ("classifier", "schema", "model", "dataset", "training", "evaluation"):
    assert name not in sys.modules, name
'''


class PackagingConfigurationTests(unittest.TestCase):
    def test_explicit_single_package_and_project_metadata(self):
        config = tomllib.loads((SOURCE / "pyproject.toml").read_text(encoding="utf-8"))
        project = config["project"]
        self.assertEqual(project["name"], "dns-exfiltration-classifier")
        self.assertEqual(project["version"], "0.1.0")
        self.assertEqual(project["requires-python"], ">=3.12")
        self.assertEqual(set(project["dependencies"]), {
            "scikit-learn==1.9.1", "numpy==2.5.4",
        })
        self.assertEqual(project["scripts"], {
            "dns-exfiltration": "dns_exfiltration_classifier.app:main",
        })
        self.assertEqual(config["tool"]["setuptools"], {
            "packages": [PACKAGE], "package-dir": {PACKAGE: "."},
            "include-package-data": False,
        })

    def test_unpickler_remaps_only_exact_artifact_globals(self):
        for module in ("model", "dns_exfiltration_classifier.model"):
            with self.subTest(module=module):
                payload = f"c{module}\nModelArtifact\n.".encode("ascii")
                self.assertIs(_ArtifactUnpickler(io.BytesIO(payload)).load(), ModelArtifact)
        self.assertIs(_ArtifactUnpickler(io.BytesIO(b"cbuiltins\ndict\n.")).load(), dict)
        with self.assertRaises(ModuleNotFoundError):
            _ArtifactUnpickler(io.BytesIO(b"cnot_an_artifact_module\nModelArtifact\n.")).load()
        with self.assertRaises(AttributeError):
            _ArtifactUnpickler(io.BytesIO(b"cmodel\nNotModelArtifact\n.")).load()


class NamespaceImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "target"
        self.target.mkdir()
        (self.target / PACKAGE).symlink_to(SOURCE, target_is_directory=True)

    def test_namespace_imports_without_trained_artifact(self):
        run_python(self, IMPORT_CHECK, cwd=self.root, arguments=(self.target, SOURCE))

    def test_runtime_only_import_does_not_load_training_or_dataset(self):
        run_python(self, RUNTIME_IMPORT_CHECK, cwd=self.root, arguments=(self.target, SOURCE))

    @unittest.skipUnless(BASELINE.is_file(), "optional trusted baseline artifact is absent")
    def test_namespace_imports_and_real_legacy_artifact_roundtrip(self):
        saved = self.root / "namespaced.pkl"
        run_python(self, ARTIFACT_CHECK, cwd=self.root, arguments=(
            self.target, BASELINE, saved,
        ))
        # A new package-qualified artifact must also load in the original flat API.
        run_python(self, '''
import sys
sys.path.insert(0, sys.argv[1])
from model import ModelArtifact, load_model
artifact = load_model(sys.argv[2])
assert type(artifact) is ModelArtifact
assert "dns_exfiltration_classifier.model" not in sys.modules
''', cwd=self.root, arguments=(SOURCE, saved))

    @unittest.skipUnless(BASELINE.is_file(), "optional trusted baseline artifact is absent")
    def test_runtime_api_loads_real_legacy_artifact_without_flat_aliases(self):
        run_python(self, RUNTIME_CHECK, cwd=self.root, arguments=(self.target, BASELINE))


class WheelInstallationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ("setuptools", "wheel", "pip")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise unittest.SkipTest("optional offline wheel tools unavailable: " + ", ".join(missing))
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        stage = cls.root / "source"
        stage.mkdir()
        for path in (*SOURCE.glob("*.py"), SOURCE / "pyproject.toml"):
            shutil.copy2(path, stage / path.name)
        # Inert stand-ins ensure explicit packaging excludes non-code trees,
        # without copying data, artifacts, or the virtual environment.
        for name in ("dataset", "dns_dataset", "runs", "bin", "tests", "venv", "lib"):
            directory = stage / name
            directory.mkdir()
            (directory / "excluded.txt").write_text("not package data\n", encoding="utf-8")
        dist = cls.root / "dist"
        dist.mkdir()
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])",
             str(dist)],
            cwd=stage, capture_output=True, text=True, timeout=120, check=False,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.wheel = next(dist.glob("*.whl"))
        cls.target = cls.root / "installed"
        result = subprocess.run(
            [sys.executable, "-I", "-m", "pip", "install", "--no-index", "--no-deps",
             "--no-compile", "--target", str(cls.target), str(cls.wheel)],
            cwd=cls.root, capture_output=True, text=True, timeout=120, check=False,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_wheel_contains_only_namespaced_modules_and_distribution_metadata(self):
        with zipfile.ZipFile(self.wheel) as archive:
            names = archive.namelist()
            for name in names:
                with self.subTest(member=name):
                    if name.startswith(PACKAGE + "/"):
                        relative = name.removeprefix(PACKAGE + "/")
                        self.assertNotIn("/", relative)
                        self.assertTrue(relative.endswith(".py"))
                    else:
                        self.assertTrue(name.split("/")[0].endswith(".dist-info"), name)
            self.assertIn(PACKAGE + "/model.py", names)
            self.assertIn(PACKAGE + "/schema.py", names)
            entry_points = next(name for name in names if name.endswith("/entry_points.txt"))
            self.assertIn(
                "dns-exfiltration = dns_exfiltration_classifier.app:main",
                archive.read(entry_points).decode("utf-8"),
            )

    def test_installed_imports_without_trained_artifact(self):
        run_python(self, IMPORT_CHECK, cwd=self.root, arguments=(self.target, SOURCE))

    def test_installed_runtime_only_import_does_not_load_training_or_dataset(self):
        run_python(self, RUNTIME_IMPORT_CHECK, cwd=self.root, arguments=(self.target, SOURCE))

    @unittest.skipUnless(BASELINE.is_file(), "optional trusted baseline artifact is absent")
    def test_installed_imports_and_real_legacy_artifact(self):
        run_python(self, ARTIFACT_CHECK, cwd=self.root, arguments=(
            self.target, BASELINE, self.root / "installed-roundtrip.pkl",
        ))

    @unittest.skipUnless(BASELINE.is_file(), "optional trusted baseline artifact is absent")
    def test_installed_runtime_matches_sklearn_on_precomputed_features(self):
        run_python(self, RUNTIME_CHECK, cwd=self.root, arguments=(self.target, BASELINE))

    @unittest.skipUnless(BASELINE.is_file(), "optional trusted baseline artifact is absent")
    def test_installed_runtime_query_callback_matches_named_features(self):
        run_python(self, QUERY_CHECK, cwd=self.root, arguments=(self.target, BASELINE))

    def test_installed_console_entry_point(self):
        run_python(self, '''
import sys
sys.path.insert(0, sys.argv[1])
from importlib.metadata import distribution
entry, = distribution("dns-exfiltration-classifier").entry_points
assert entry.name == "dns-exfiltration"
assert entry.value == "dns_exfiltration_classifier.app:main"
try:
    entry.load()(["--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("--help did not exit")
''', cwd=self.root, arguments=(self.target,))


if __name__ == "__main__":
    unittest.main()
