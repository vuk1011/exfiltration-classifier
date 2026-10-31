"""Tests for the dataset summary command."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from app import main
from schema import FEATURE_NAMES


class SummaryCommandTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "dataset.csv"
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for label in ("True", "False", "True"):
                writer.writerow(
                    ["192.0.2.1", "example.test", "123", label, "x.example.test"]
                    + [1] * 9 + [0] * 8
                )

    def test_summary_json_includes_source_and_sample_counts(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main([
                "summary", str(self.path), "--sample-size", "2", "--seed", "42",
                "--include-metadata",
            ])
        summary = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(summary["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(summary["source_counts"], {"benign": 1, "attack": 2, "total": 3})
        self.assertEqual(summary["selected_counts"]["total"], 2)
        self.assertEqual(summary["sampled_out_rows"], 1)
        self.assertEqual(summary["excluded_rows"], 0)
        self.assertEqual(summary["seed"], 42)
        self.assertTrue(summary["include_metadata"])

    def test_default_sample_is_bounded(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["summary", str(self.path)])
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["sample_size"], 10_000)
        self.assertEqual(summary["sampling_method"], "reservoir")

    def test_all_retains_every_record(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["summary", str(self.path), "--all"])
        summary = json.loads(output.getvalue())
        self.assertIsNone(summary["sample_size"])
        self.assertEqual(summary["sampling_method"], "all")
        self.assertEqual(summary["selected_counts"], summary["source_counts"])

    def test_invalid_dataset_and_arguments_exit_with_actionable_errors(self):
        cases = (
            (["summary", str(self.path), "--sample-size", "0"], "positive integer"),
            (["summary", str(self.path), "--sample-size", "abc"], "invalid int"),
            (["summary", str(self.path), "--all", "--sample-size", "1"], "not allowed"),
            (["summary", str(self.path.with_name("missing.csv"))], "missing.csv"),
        )
        for arguments, message in cases:
            output = io.StringIO()
            errors = io.StringIO()
            with (
                self.subTest(arguments=arguments),
                redirect_stdout(output),
                redirect_stderr(errors),
                self.assertRaises(SystemExit) as raised,
            ):
                main(arguments)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, errors.getvalue())
            self.assertEqual(output.getvalue(), "")

    def test_invalid_record_does_not_emit_partial_summary(self):
        self.path.write_text("invalid\n", encoding="utf-8")
        output = io.StringIO()
        errors = io.StringIO()
        with (
            redirect_stdout(output), redirect_stderr(errors),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["summary", str(self.path)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("record 1", errors.getvalue())
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
