"""Portable split manifests tested against tiny synthetic, headerless CSVs."""

import copy
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset import DatasetError, load_dataset
from schema import FEATURE_NAMES
from splits import (
    SPLIT_NAMES,
    create_splits,
    describe_source,
    restore_splits,
    validate_split_settings,
)


class SplitSettingsTests(unittest.TestCase):
    def test_valid_settings_and_seed_boundaries(self):
        for seed in (0, 2**32 - 1):
            with self.subTest(seed=seed):
                self.assertIsNone(validate_split_settings(seed, 0.1, 0.1))
        self.assertIsNone(validate_split_settings(17, 0.2, 0.3))

    def test_invalid_seed(self):
        for seed in (True, False, 1.5, "0", None, -1, 2**32):
            error = TypeError if not isinstance(seed, int) or isinstance(seed, bool) else ValueError
            with self.subTest(seed=seed), self.assertRaisesRegex(error, "seed"):
                validate_split_settings(seed, 0.1, 0.1)  # pyright: ignore[reportArgumentType]

    def test_invalid_fractions(self):
        for name in ("validation_size", "test_size"):
            for value in (True, False, None, "0.1", 0, -0.1, 1, 2, float("nan"), float("inf"), -float("inf")):
                settings = {"validation_size": 0.1, "test_size": 0.1, name: value}
                error = TypeError if isinstance(value, bool) or not isinstance(value, (int, float)) else ValueError
                with self.subTest(name=name, value=value), self.assertRaisesRegex(error, name):
                    validate_split_settings(0, **settings)
        for fractions in ((0.5, 0.5), (0.6, 0.5)):
            with self.subTest(fractions=fractions), self.assertRaisesRegex(ValueError, "less than 1"):
                validate_split_settings(0, *fractions)


class SplitManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "source.csv"
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(
                ["192.0.2.1", "example.test", "123", str(i > 50), f"q{i}.example.test",
                 *[i + j / 10 for j in range(9)], *["ignored"] * 8]
                for i in range(1, 101)
            )
        self.dataset = load_dataset(self.path, sample_size=None, seed=0, include_metadata=True)
        self.source = describe_source(self.path)
        self.partitions, self.manifest = create_splits(self.dataset, source=self.source)
        self.manifest_path = self.directory / "splits.json"

    def write_manifest(self, manifest=None):
        self.manifest_path.write_text(
            json.dumps(self.manifest if manifest is None else manifest), encoding="utf-8"
        )
        return self.manifest_path

    def restore(self, *, dataset=None, source=None, **settings):
        return restore_splits(
            self.manifest_path,
            self.dataset if dataset is None else dataset,
            source=self.source if source is None else source,
            **settings,
        )

    def test_describe_source_uses_stat_without_opening_or_reading_contents(self):
        stat = self.path.stat()
        with (
            patch.object(Path, "open", side_effect=AssertionError("must not open source")) as opened,
            patch("builtins.open", side_effect=AssertionError("must not read source")),
            patch("os.open", side_effect=AssertionError("must not open source descriptor")),
        ):
            source = describe_source(self.path)
        opened.assert_not_called()
        self.assertEqual(source, {
            "path": str(self.path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
        self.assertEqual(source, self.source)

    def test_same_path_size_and_mtime_cannot_detect_content_edits(self):
        self.write_manifest()
        stat = self.path.stat()
        original = self.path.read_bytes()
        changed = original.replace(b"q1.example", b"x1.example", 1)
        self.assertNotEqual(changed, original)
        self.path.write_bytes(changed)
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        source = describe_source(self.path)
        self.assertEqual(source, self.source)
        # Metadata is not integrity verification: preserving these fields hides edits.
        dataset = load_dataset(self.path, seed=0, include_metadata=True)
        restored, manifest = self.restore(dataset=dataset, source=source)
        self.assertEqual(manifest["assignments"], self.manifest["assignments"])
        self.assertEqual(sum(map(len, restored.values())), 100)

    def test_stratified_80_10_10_has_canonical_disjoint_complete_record_ids(self):
        self.assertEqual(tuple(self.partitions), SPLIT_NAMES)
        self.assertEqual(self.manifest["format_version"], 2)
        self.assertEqual(self.manifest["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(self.manifest["source"], self.source)
        self.assertEqual(self.manifest["sampling"], {"sample_size": None, "seed": 0})
        self.assertEqual(self.manifest["split_settings"], {
            "seed": 0, "validation_size": 0.1, "test_size": 0.1,
        })
        all_ids = []
        for name, size in zip(SPLIT_NAMES, (80, 10, 10), strict=True):
            rows = self.partitions[name]
            records = [row.row_number for row in rows]
            self.assertEqual(len(rows), size)
            self.assertEqual({row.attack for row in rows}, {False, True})
            self.assertEqual(sum(row.attack for row in rows), size // 2)
            self.assertEqual(records, sorted(records))
            self.assertEqual(self.manifest["assignments"][name], records)
            all_ids.extend(records)
        self.assertEqual(sorted(all_ids), list(range(1, 101)))

    def test_repeat_and_json_restore_are_identical_without_resplitting(self):
        repeated, manifest = create_splits(self.dataset, source=self.source)
        self.assertEqual(repeated, self.partitions)
        self.assertEqual(manifest, self.manifest)
        self.write_manifest()
        with patch("splits.train_test_split", side_effect=AssertionError("must not re-split")):
            restored, manifest = self.restore()
        self.assertEqual(restored, self.partitions)
        self.assertEqual(manifest, self.manifest)

    def test_restore_sorts_record_ids_into_source_order(self):
        manifest = copy.deepcopy(self.manifest)
        for records in manifest["assignments"].values():
            records.reverse()
        self.write_manifest(manifest)
        restored, _ = self.restore()
        self.assertEqual(restored, self.partitions)

    def test_same_bytes_at_new_path_are_rejected_even_with_matching_mtime(self):
        relocated = self.directory / "relocated.csv"
        relocated.write_bytes(self.path.read_bytes())
        stat = self.path.stat()
        os.utime(relocated, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.write_manifest()
        dataset = load_dataset(relocated, seed=0, include_metadata=True)
        source = describe_source(relocated)
        self.assertEqual(source["size_bytes"], self.source["size_bytes"])
        self.assertEqual(source["mtime_ns"], self.source["mtime_ns"])
        with self.assertRaisesRegex(DatasetError, "source"):
            self.restore(dataset=dataset, source=source)

    def test_same_size_mutation_with_changed_mtime_and_wrong_size_are_rejected(self):
        self.write_manifest()
        stat = self.path.stat()
        original = self.path.read_bytes()
        self.path.write_bytes(original.replace(b"q1.example", b"x1.example", 1))
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        source = describe_source(self.path)
        self.assertEqual(source["size_bytes"], len(original))
        self.assertNotEqual(source["mtime_ns"], self.source["mtime_ns"])
        with self.assertRaisesRegex(DatasetError, "source"):
            self.restore(source=source)
        wrong_size = {**self.source, "size_bytes": self.source["size_bytes"] + 1}
        with self.assertRaisesRegex(DatasetError, "source"):
            self.restore(source=wrong_size)

    def test_reuse_rejects_changed_sampling_fractions_and_seed(self):
        self.write_manifest()
        # Even a sample retaining every row must preserve the requested sampling config.
        for size in (80, 100):
            sampled = load_dataset(self.path, sample_size=size, seed=0)
            with self.subTest(sample_size=size), self.assertRaisesRegex(DatasetError, "sampling settings"):
                self.restore(dataset=sampled)
        with self.assertRaisesRegex(DatasetError, "sampling settings"):
            self.restore(dataset=load_dataset(self.path, seed=1), seed=1)
        for settings in ({"seed": 1}, {"validation_size": 0.2}, {"test_size": 0.2}):
            with self.subTest(settings=settings), self.assertRaisesRegex(DatasetError, "split settings"):
                self.restore(**settings)

    def test_malformed_manifest_header_and_schema(self):
        cases: list[tuple[object, str]] = [([], "JSON object")]
        for field, value, message in (
            ("format_version", True, "format_version"),
            ("format_version", 1, "format_version"),
            ("format_version", 3, "format_version"),
            ("feature_names", list(reversed(FEATURE_NAMES)), "feature schema"),
            ("source", None, "source"),
            ("assignments", [], "must assign"),
            ("assignments", {"train": []}, "must assign"),
        ):
            cases.append(({**self.manifest, field: value}, message))
        for manifest, message in cases:
            with self.subTest(manifest=manifest):
                self.write_manifest(manifest)
                with self.assertRaisesRegex(DatasetError, message):
                    self.restore()

    def test_duplicate_overlap_outside_and_incomplete_assignments_are_rejected(self):
        for kind, message in (
            ("duplicate", "duplicate or overlapping"),
            ("overlap", "duplicate or overlapping"),
            ("outside", "outside the selected sample"),
            ("incomplete", "complete selected sample"),
            ("bool", "integer lists"),
            ("string", "integer lists"),
            ("not-list", "integer lists"),
        ):
            manifest = copy.deepcopy(self.manifest)
            assignments = manifest["assignments"]
            if kind == "duplicate":
                assignments["train"].append(assignments["train"][0])
            elif kind == "overlap":
                assignments["test"].append(assignments["train"][0])
            elif kind == "outside":
                assignments["test"][0] = 101
            elif kind == "incomplete":
                assignments["train"].pop()
            elif kind == "not-list":
                assignments["train"] = "1,2,3"
            else:
                assignments["train"][0] = True if kind == "bool" else "1"
            with self.subTest(kind=kind):
                self.write_manifest(manifest)
                with self.assertRaisesRegex(DatasetError, message):
                    self.restore()

    def test_record_in_source_but_outside_reservoir_sample_is_rejected(self):
        sampled = load_dataset(self.path, sample_size=60, seed=0)
        _, manifest = create_splits(sampled, source=self.source)
        selected = {row.row_number for row in sampled.rows}
        manifest["assignments"]["test"][0] = next(i for i in range(1, 101) if i not in selected)
        self.write_manifest(manifest)
        with self.assertRaisesRegex(DatasetError, "outside the selected sample"):
            self.restore(dataset=sampled)

    def test_restored_partition_must_still_have_both_classes(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["assignments"] = {
            "train": list(range(1, 81)), "validation": list(range(81, 91)),
            "test": list(range(91, 101)),
        }
        self.write_manifest(manifest)
        with self.assertRaisesRegex(DatasetError, "validation split requires both classes"):
            self.restore()

    def test_insufficient_minority_has_actionable_error(self):
        # The sole attack is retained, so this exercises stratification, not single-class input.
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(
                ["ip", "domain", "time", str(i == 100), "query", *[i] * 9, *[0] * 8]
                for i in range(1, 101)
            )
        dataset = load_dataset(self.path)
        with self.assertRaisesRegex(DatasetError, "increase the sample size or adjust split fractions"):
            create_splits(dataset, source=describe_source(self.path))


if __name__ == "__main__":
    unittest.main()
