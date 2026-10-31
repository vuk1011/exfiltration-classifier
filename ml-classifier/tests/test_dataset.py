"""Dataset contract tests using only synthetic CSVs and the stdlib."""

import csv
import random
import tempfile
import unittest
from pathlib import Path

from dataset import Dataset, DatasetError, iter_rows, load_dataset
from schema import (
    AGGREGATE_COLUMNS,
    CSV_COLUMNS,
    FEATURE_NAMES,
    FEATURE_START,
    FEATURE_STOP,
    LABEL_INDEX,
    METADATA_COLUMNS,
    ClassCounts,
    DatasetRow,
    RequestMetadata,
)


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "dataset.csv"

    def make_row(self, number=1, attack=False):
        return [
            "192.0.2.1", "example.test", "2026-01-01T00:00:00Z",
            "True" if attack else "False", f"request-{number}.example.test",
            *[str(number + index / 10) for index in range(9)],
            *["ignored" for _ in range(8)],
        ]

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        return self.path

    def write_mixed_rows(self, count=60):
        return self.write_rows(
            self.make_row(number, attack=number % 3 == 0)
            for number in range(1, count + 1)
        )

    def assert_bad_record(self, fields, message):
        self.write_rows([self.make_row(), fields])
        for read in (lambda path: tuple(iter_rows(path)), load_dataset):
            with self.subTest(reader=read):
                with self.assertRaises(DatasetError) as caught:
                    read(self.path)
                error = str(caught.exception)
                self.assertIn(f"{self.path.resolve()}: record 2", error)
                self.assertIn(message, error)


class SchemaTests(unittest.TestCase):
    def test_published_column_names_and_order(self):
        self.assertEqual(METADATA_COLUMNS, (
            "user_ip", "domain", "timestamp", "attack", "request",
        ))
        self.assertEqual(FEATURE_NAMES, (
            "len", "subdomains_count", "w_count", "w_max", "entropy",
            "w_max_ratio", "w_count_ratio", "digits_ratio", "uppercase_ratio",
        ))
        self.assertEqual(AGGREGATE_COLUMNS, (
            "time_avg", "time_stdev", "size_avg", "size_stdev",
            "throughput", "unique", "entropy_avg", "entropy_stdev",
        ))
        self.assertEqual(len(CSV_COLUMNS), 22)
        self.assertEqual(CSV_COLUMNS, METADATA_COLUMNS + FEATURE_NAMES + AGGREGATE_COLUMNS)
        self.assertEqual((FEATURE_START, FEATURE_STOP, LABEL_INDEX), (5, 14, 3))
        self.assertEqual(CSV_COLUMNS[5:14], FEATURE_NAMES)

    def test_class_counts_total(self):
        self.assertEqual(ClassCounts().total, 0)
        self.assertEqual(ClassCounts(benign=7, attack=3).total, 10)


class IterRowsTests(DatasetTestCase):
    def test_headerless_rows_features_labels_and_source_order(self):
        self.write_rows([self.make_row(7, True), self.make_row(2, False)])
        rows = tuple(iter_rows(self.path))
        self.assertEqual(len(rows), 2)
        for row, number, attack, record in zip(rows, (7, 2), (True, False), (1, 2)):
            with self.subTest(record=record):
                self.assertIsInstance(row, DatasetRow)
                self.assertEqual(row.features, tuple(number + index / 10 for index in range(9)))
                self.assertTrue(all(isinstance(value, float) for value in row.features))
                self.assertIs(row.attack, attack)
                self.assertEqual(row.row_number, record)
                self.assertEqual(row.source, str(self.path.resolve()))
                self.assertIsNone(row.metadata)

    def test_source_is_resolved_from_string_path_with_parent_segments(self):
        self.write_rows([self.make_row()])
        nested = self.directory / "nested"
        nested.mkdir()
        unresolved = str(nested / ".." / self.path.name)
        row = next(iter_rows(unresolved))
        self.assertEqual(row.source, str(self.path.resolve()))

    def test_metadata_preserves_quoted_commas_quotes_and_newlines(self):
        fields = self.make_row()
        fields[4] = 'query,"quoted",part\nsecond physical line.example.test'
        self.write_rows([fields, self.make_row(2, True)])
        rows = tuple(iter_rows(self.path, include_metadata=True))
        self.assertEqual(rows[0].metadata, RequestMetadata(
            user_ip=fields[0], domain=fields[1], timestamp=fields[2], request=fields[4],
        ))
        self.assertEqual([row.row_number for row in rows], [1, 2])
        assert rows[1].metadata is not None
        self.assertEqual(rows[1].metadata.request, "request-2.example.test")
        self.assertEqual(
            [row.features for row in rows],
            [row.features for row in iter_rows(self.path)],
        )

    def test_aggregate_values_and_nonlabel_metadata_are_not_numeric_inputs(self):
        fields = self.make_row()
        fields[:3] = ["", "not a domain", "not a timestamp"]
        fields[4] = ""
        fields[14:] = ["", "NaN", "inf", "-inf", "not numeric", "1e9999", "True", "a,b"]
        self.write_rows([fields])
        row = next(iter_rows(self.path, include_metadata=True))
        self.assertEqual(row.features, tuple(1 + index / 10 for index in range(9)))
        assert row.metadata is not None
        self.assertEqual(row.metadata.domain, "not a domain")

    def test_finite_numeric_formats_are_accepted(self):
        fields = self.make_row()
        fields[5:14] = ["0", "-2", "+3", "4.5", "1e2", "-2E-2", " 6 ", ".7", "8."]
        self.write_rows([fields])
        self.assertEqual(next(iter_rows(self.path)).features,
                         (0.0, -2.0, 3.0, 4.5, 100.0, -0.02, 6.0, 0.7, 8.0))

    def test_invalid_column_counts_have_record_location(self):
        for count in (0, 1, 5, 14, 21, 23, 30):
            with self.subTest(columns=count):
                fields = (self.make_row() + ["extra"] * 8)[:count]
                self.assert_bad_record(fields, f"expected 22 columns in a headerless CSV, got {count}")

    def test_header_row_is_rejected(self):
        self.write_rows([CSV_COLUMNS])
        with self.assertRaises(DatasetError) as caught:
            tuple(iter_rows(self.path))
        self.assertIn(f"{self.path.resolve()}: record 1", str(caught.exception))
        self.assertIn("attack must be", str(caught.exception))

    def test_only_exact_boolean_label_spellings_are_accepted(self):
        for label in ("true", "false", "TRUE", "FALSE", "1", "0", "yes", "no", "", " True", "False "):
            with self.subTest(label=label):
                fields = self.make_row()
                fields[3] = label
                self.assert_bad_record(fields, "attack must be 'True' or 'False'")

    def test_each_feature_rejects_nonnumeric_values_with_feature_name(self):
        for index, name in enumerate(FEATURE_NAMES, start=5):
            for value in ("", "not-a-number"):
                with self.subTest(feature=name, value=value):
                    fields = self.make_row()
                    fields[index] = value
                    self.assert_bad_record(fields, f"{name} must be numeric")

    def test_each_feature_rejects_nonfinite_values_with_feature_name(self):
        for index, name in enumerate(FEATURE_NAMES, start=5):
            for value in ("nan", "NaN", "inf", "+Infinity", "-inf", "1e9999", "-1e9999"):
                with self.subTest(feature=name, value=value):
                    fields = self.make_row()
                    fields[index] = value
                    self.assert_bad_record(fields, f"{name} must be finite")

    def test_empty_file_is_rejected_by_both_entry_points(self):
        self.write_rows([])
        for read in (lambda path: tuple(iter_rows(path)), load_dataset):
            with self.subTest(reader=read):
                with self.assertRaises(DatasetError) as caught:
                    read(self.path)
                self.assertIn(str(self.path.resolve()), str(caught.exception))
                self.assertIn("empty", str(caught.exception))

    def test_malformed_csv_is_wrapped_with_source_location(self):
        self.path.write_text('"unterminated field', encoding="utf-8")
        with self.assertRaises(DatasetError) as caught:
            tuple(iter_rows(self.path))
        self.assertIn(str(self.path.resolve()), str(caught.exception))
        self.assertIn("physical line 1", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, csv.Error)

    def test_invalid_utf8_is_rejected(self):
        self.path.write_bytes(b"\xff\xfe")
        with self.assertRaises(DatasetError) as caught:
            tuple(iter_rows(self.path))
        self.assertIn(str(self.path.resolve()), str(caught.exception))
        self.assertIn("UTF-8", str(caught.exception))


class LoadDatasetTests(DatasetTestCase):
    def test_default_load_retains_every_row_in_source_order(self):
        self.write_mixed_rows(12)
        dataset = load_dataset(self.path)
        self.assertIsInstance(dataset, Dataset)
        self.assertIsInstance(dataset.rows, tuple)
        self.assertEqual(dataset.rows, tuple(iter_rows(self.path)))
        self.assertEqual([row.row_number for row in dataset.rows], list(range(1, 13)))
        summary = dataset.summary
        self.assertEqual(summary.source, str(self.path.resolve()))
        self.assertEqual(summary.feature_names, FEATURE_NAMES)
        self.assertEqual(summary.source_counts, ClassCounts(benign=8, attack=4))
        self.assertEqual(summary.selected_counts, summary.source_counts)
        self.assertIsNone(summary.sample_size)
        self.assertEqual(summary.seed, 0)
        self.assertIs(summary.include_metadata, False)
        self.assertEqual(summary.sampling_method, "all")
        self.assertEqual(summary.sampled_out_rows, 0)
        self.assertEqual(summary.excluded_rows, 0)

    def test_explicit_none_and_metadata_option(self):
        self.write_mixed_rows(5)
        dataset = load_dataset(str(self.path), sample_size=None, seed=-8, include_metadata=True)
        self.assertEqual(dataset.rows, tuple(iter_rows(self.path, include_metadata=True)))
        self.assertTrue(all(isinstance(row.metadata, RequestMetadata) for row in dataset.rows))
        self.assertIs(dataset.summary.include_metadata, True)
        self.assertEqual(dataset.summary.seed, -8)
        self.assertEqual(dataset.summary.sampling_method, "all")

    def test_seeded_reservoir_uses_full_source_and_restores_record_order(self):
        self.write_mixed_rows()
        dataset = load_dataset(self.path, sample_size=7, seed=42)
        record_numbers = [row.row_number for row in dataset.rows]
        self.assertEqual(record_numbers, [10, 12, 18, 36, 47, 53, 59])
        self.assertEqual(record_numbers, sorted(record_numbers))
        self.assertEqual(len(set(record_numbers)), 7)
        self.assertTrue(any(number > 7 for number in record_numbers))
        source_rows = tuple(iter_rows(self.path))
        self.assertEqual(dataset.rows, tuple(source_rows[number - 1] for number in record_numbers))

    def test_sample_summary_distinguishes_source_selected_and_sampled_out(self):
        self.write_mixed_rows()
        dataset = load_dataset(self.path, sample_size=7, seed=42, include_metadata=True)
        summary = dataset.summary
        self.assertEqual(summary.source, str(self.path.resolve()))
        self.assertEqual(summary.feature_names, FEATURE_NAMES)
        self.assertEqual(summary.source_counts, ClassCounts(benign=40, attack=20))
        self.assertEqual(summary.source_counts.total, 60)
        self.assertEqual(summary.selected_counts, ClassCounts(benign=4, attack=3))
        self.assertEqual(summary.selected_counts.total, len(dataset.rows))
        self.assertEqual(summary.sample_size, 7)
        self.assertEqual(summary.seed, 42)
        self.assertIs(summary.include_metadata, True)
        self.assertEqual(summary.sampling_method, "reservoir")
        self.assertEqual(summary.sampled_out_rows, 53)
        self.assertEqual(summary.excluded_rows, 0)
        self.assertTrue(all(row.metadata is not None for row in dataset.rows))

    def test_same_seed_reproduces_sample_and_different_seed_changes_it(self):
        self.write_mixed_rows()
        first = load_dataset(self.path, sample_size=7, seed=42)
        self.assertEqual(first, load_dataset(self.path, sample_size=7, seed=42))
        self.assertNotEqual(first.rows, load_dataset(self.path, sample_size=7, seed=43).rows)

    def test_sampling_is_independent_of_and_does_not_mutate_global_random(self):
        self.write_mixed_rows()
        initial_state = random.getstate()
        self.addCleanup(random.setstate, initial_state)
        samples = []
        for global_seed in (123, 987):
            random.seed(global_seed)
            for _ in range(13):
                random.random()
            before = random.getstate()
            samples.append(load_dataset(self.path, sample_size=7, seed=42))
            self.assertEqual(random.getstate(), before)
        self.assertEqual(samples[0], samples[1])

    def test_equal_and_oversized_samples_retain_all_rows(self):
        self.write_mixed_rows(6)
        expected = tuple(iter_rows(self.path))
        for size in (6, 7, 100):
            with self.subTest(sample_size=size):
                dataset = load_dataset(self.path, sample_size=size)
                self.assertEqual(dataset.rows, expected)
                self.assertEqual(dataset.summary.sample_size, size)
                self.assertEqual(dataset.summary.sampling_method, "reservoir")
                self.assertEqual(dataset.summary.source_counts, ClassCounts(4, 2))
                self.assertEqual(dataset.summary.selected_counts, ClassCounts(4, 2))
                self.assertEqual(dataset.summary.sampled_out_rows, 0)
                self.assertEqual(dataset.summary.excluded_rows, 0)

    def test_single_row_sample_is_allowed_but_not_trainable(self):
        self.write_mixed_rows()
        dataset = load_dataset(self.path, sample_size=1, seed=42)
        self.assertEqual(len(dataset.rows), 1)
        self.assertEqual(dataset.summary.source_counts.total, 60)
        self.assertEqual(dataset.summary.sampled_out_rows, 59)
        with self.assertRaisesRegex(DatasetError, "both benign and attack"):
            dataset.validate_for_training()

    def test_malformed_late_unselected_records_are_still_validated(self):
        self.write_mixed_rows()
        baseline = load_dataset(self.path, sample_size=7, seed=42)
        self.assertNotIn(60, [row.row_number for row in baseline.rows])
        invalid_rows = []
        fields = self.make_row(60, True)
        invalid_rows.append((fields[:-1], "expected 22 columns"))
        fields = self.make_row(60, True)
        fields[3] = "true"
        invalid_rows.append((fields, "attack must be"))
        fields = self.make_row(60, True)
        fields[5] = "bad-number"
        invalid_rows.append((fields, "len must be numeric"))
        fields = self.make_row(60, True)
        fields[13] = "NaN"
        invalid_rows.append((fields, "uppercase_ratio must be finite"))
        for invalid, message in invalid_rows:
            with self.subTest(error=message):
                rows = [self.make_row(number, number % 3 == 0) for number in range(1, 60)]
                self.write_rows([*rows, invalid])
                with self.assertRaises(DatasetError) as caught:
                    load_dataset(self.path, sample_size=7, seed=42)
                self.assertIn(f"{self.path.resolve()}: record 60", str(caught.exception))
                self.assertIn(message, str(caught.exception))

    def test_single_class_inputs_are_valid_for_evaluation_not_training(self):
        for attack in (True, False):
            with self.subTest(attack=attack):
                self.write_rows(self.make_row(number, attack) for number in range(1, 6))
                dataset = load_dataset(self.path)
                expected = ClassCounts(0, 5) if attack else ClassCounts(5, 0)
                self.assertEqual(dataset.summary.source_counts, expected)
                self.assertEqual(dataset.summary.selected_counts, expected)
                self.assertEqual(len(dataset.rows), 5)
                self.assertTrue(all(row.attack is attack for row in dataset.rows))
                with self.assertRaisesRegex(DatasetError, "both benign and attack"):
                    dataset.validate_for_training()

    def test_mixed_selected_classes_are_valid_for_training(self):
        self.write_mixed_rows()
        for size in (None, 7):
            with self.subTest(sample_size=size):
                dataset = load_dataset(self.path, sample_size=size, seed=42)
                self.assertIsNone(dataset.validate_for_training())

    def test_nonpositive_sample_sizes_are_rejected(self):
        self.write_mixed_rows(3)
        for size in (0, -1, -100):
            with self.subTest(sample_size=size), self.assertRaisesRegex(ValueError, "sample_size"):
                load_dataset(self.path, sample_size=size)

    def test_invalid_sample_size_types_are_rejected(self):
        self.write_mixed_rows(3)
        for size in (True, False, 1.0, 2.5, "2", "", [], {}):
            with self.subTest(sample_size=size), self.assertRaisesRegex(TypeError, "sample_size"):
                load_dataset(self.path, sample_size=size)  # pyright: ignore[reportArgumentType]

    def test_invalid_seeds_are_rejected(self):
        self.write_mixed_rows(3)
        for seed in (None, True, False, 1.0, 2.5, "42", "", [], {}):
            with self.subTest(seed=seed), self.assertRaisesRegex(TypeError, "seed"):
                load_dataset(self.path, seed=seed)  # pyright: ignore[reportArgumentType]

    def test_zero_negative_and_large_integer_seeds_are_accepted(self):
        self.write_mixed_rows(6)
        for seed in (0, -1, 2**128):
            with self.subTest(seed=seed):
                dataset = load_dataset(self.path, sample_size=2, seed=seed)
                self.assertEqual(len(dataset.rows), 2)
                self.assertEqual(dataset.summary.seed, seed)
                self.assertEqual(dataset, load_dataset(self.path, sample_size=2, seed=seed))


if __name__ == "__main__":
    unittest.main()
