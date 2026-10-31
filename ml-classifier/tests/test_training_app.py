"""CLI contracts for the lazily imported training workflow."""

import io
import json
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from app import main
from dataset import DatasetError


class TrainingModule(types.ModuleType):
    train_baseline: Mock
    evaluate_file: Mock
    write_report: Mock


class TrainingCommandTests(unittest.TestCase):
    def setUp(self):
        self.report = {"cohort": "test", "metrics": {"precision": None}}
        self.training = TrainingModule("training")
        self.training.train_baseline = Mock(return_value=self.report)
        self.training.evaluate_file = Mock(return_value=self.report)
        self.training.write_report = Mock()
        self.modules = patch.dict(sys.modules, {"training": self.training})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def run_command(self, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(arguments)
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), self.report)

    def assert_error(self, arguments, message):
        output, errors = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(output), redirect_stderr(errors),
            self.assertRaises(SystemExit) as raised,
        ):
            main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn(message, errors.getvalue())

    def test_train_defaults(self):
        self.run_command(["train", "input.csv", "--output-dir", "run"])
        self.training.train_baseline.assert_called_once_with(
            Path("input.csv"), Path("run"), sample_size=10_000, seed=0,
            validation_size=0.1, test_size=0.1, reuse_splits=None, error_limit=10,
        )
        self.training.evaluate_file.assert_not_called()
        self.training.write_report.assert_not_called()

    def test_train_custom_options_and_reused_assignments(self):
        self.run_command([
            "train", "input.csv", "--output-dir", "new-run", "--sample-size", "200",
            "--seed", "42", "--validation-size", ".2", "--test-size", ".3",
            "--reuse-splits", "old-run/splits.json", "--error-limit", "0",
        ])
        self.training.train_baseline.assert_called_once_with(
            Path("input.csv"), Path("new-run"), sample_size=200, seed=42,
            validation_size=0.2, test_size=0.3,
            reuse_splits=Path("old-run/splits.json"), error_limit=0,
        )

    def test_evaluate_defaults_and_report_saved_before_stdout(self):
        def check_report(path, report):
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(path, Path("external.json"))
            self.assertIs(report, self.report)

        output = io.StringIO()
        self.training.write_report.side_effect = check_report
        with redirect_stdout(output):
            result = main([
                "evaluate", "input.csv", "--model", "run/model.pkl",
                "--report", "external.json",
            ])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), self.report)
        self.training.evaluate_file.assert_called_once_with(
            Path("input.csv"), Path("run/model.pkl"), sample_size=10_000,
            seed=0, cohort="external", error_limit=10,
        )
        self.training.write_report.assert_called_once_with(
            Path("external.json"), self.report
        )
        self.training.train_baseline.assert_not_called()

    def test_evaluate_custom_options(self):
        self.run_command([
            "evaluate", "input.csv", "--model", "run/model.pkl",
            "--report", "report.json", "--cohort", "modified", "--sample-size", "50",
            "--seed", "17", "--error-limit", "3",
        ])
        self.training.evaluate_file.assert_called_once_with(
            Path("input.csv"), Path("run/model.pkl"), sample_size=50,
            seed=17, cohort="modified", error_limit=3,
        )

    def test_all_passes_none_for_each_workflow(self):
        for command, options, function in (
            ("train", ["--output-dir", "run"], self.training.train_baseline),
            ("evaluate", ["--model", "model.pkl", "--report", "report.json"],
             self.training.evaluate_file),
        ):
            with self.subTest(command=command):
                self.run_command([command, "input.csv", *options, "--all"])
                self.assertIsNone(function.call_args.kwargs["sample_size"])

    def test_required_options_and_argument_parsing(self):
        cases = (
            (["train", "input.csv"], "--output-dir"),
            (["evaluate", "input.csv", "--report", "report.json"], "--model"),
            (["evaluate", "input.csv", "--model", "model.pkl"], "--report"),
            (["train", "input.csv", "--output-dir", "run", "--all",
              "--sample-size", "2"], "not allowed"),
            (["evaluate", "input.csv", "--model", "model.pkl", "--report", "r.json",
              "--all", "--sample-size", "2"], "not allowed"),
            (["train", "input.csv", "--output-dir", "run", "--sample-size", "bad"],
             "invalid int"),
            (["train", "input.csv", "--output-dir", "run", "--validation-size", "bad"],
             "invalid float"),
            (["evaluate", "input.csv", "--model", "model.pkl", "--report", "r.json",
              "--error-limit", "bad"], "invalid int"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                self.assert_error(arguments, message)
        self.training.train_baseline.assert_not_called()
        self.training.evaluate_file.assert_not_called()
        self.training.write_report.assert_not_called()

    def test_workflow_validation_and_io_errors_produce_no_json(self):
        for command, options, function in (
            ("train", ["--output-dir", "run"], self.training.train_baseline),
            ("evaluate", ["--model", "model.pkl", "--report", "report.json"],
             self.training.evaluate_file),
        ):
            for error in (
                DatasetError("invalid CSV record"), ValueError("insufficient minority rows"),
                FileExistsError("output already exists"), OSError("cannot read source"),
            ):
                with self.subTest(command=command, error=error):
                    function.side_effect = error
                    self.assert_error([command, "input.csv", *options], str(error))
            function.side_effect = None
        self.training.write_report.assert_not_called()

    def test_report_write_failures_produce_no_stdout(self):
        for error in (FileExistsError("report exists"), FileNotFoundError("parent missing")):
            with self.subTest(error=error):
                self.training.write_report.side_effect = error
                self.assert_error([
                    "evaluate", "input.csv", "--model", "model.pkl",
                    "--report", "report.json",
                ], str(error))

    def test_missing_training_or_dependency_has_install_hint(self):
        with patch.dict(sys.modules, {"training": None}):
            for arguments in (
                ["train", "input.csv", "--output-dir", "run"],
                ["evaluate", "input.csv", "--model", "model.pkl", "--report", "r.json"],
            ):
                with self.subTest(arguments=arguments):
                    self.assert_error(arguments, "./bin/pip install -r requirements.txt")
        self.training.train_baseline.side_effect = ModuleNotFoundError("No module named sklearn")
        self.assert_error(["train", "input.csv", "--output-dir", "run"], "sklearn")

    def test_unexpected_bugs_are_not_swallowed(self):
        for error in (RuntimeError("unexpected bug"), TypeError("programming error")):
            with self.subTest(error=error), redirect_stdout(io.StringIO()):
                self.training.train_baseline.side_effect = error
                with self.assertRaises(type(error)):
                    main(["train", "input.csv", "--output-dir", "run"])

    def test_import_and_summary_without_site_packages_or_training(self):
        script = '''
import builtins
import csv
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
original_import = builtins.__import__
def stdlib_only(name, *args, **kwargs):
    if name.split(".")[0] in {"training", "sklearn", "numpy", "model", "evaluation", "splits"}:
        raise AssertionError("unexpected ML import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = stdlib_only
from app import main
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "input.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["ip", "domain", "time", "True", "query"] + [1] * 9 + [0] * 8)
    output = io.StringIO()
    with redirect_stdout(output):
        assert main(["summary", str(path)]) == 0
    assert json.loads(output.getvalue())["selected_counts"]["total"] == 1
'''
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
