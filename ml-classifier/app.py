"""Summarize DNS datasets, train a baseline, and evaluate trusted models."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dataset import DatasetError, load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary_parser = commands.add_parser(
        "summary", help="Validate a complete CSV and summarize a reproducible sample"
    )
    summary_parser.add_argument("path", type=Path, help="Published headerless CSV")
    sampling = summary_parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-size", type=int, default=10_000)
    sampling.add_argument(
        "--all", action="store_true", help="Retain every row (may exhaust memory)"
    )
    summary_parser.add_argument("--seed", type=int, default=0)
    summary_parser.add_argument("--include-metadata", action="store_true")
    train_parser = commands.add_parser("train", help="Train and save a baseline model")
    train_parser.add_argument("path", type=Path, help="Published headerless CSV")
    train_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory for model.pkl, splits.json, report.json",
    )
    train_sampling = train_parser.add_mutually_exclusive_group()
    train_sampling.add_argument(
        "--sample-size", type=int, default=10_000,
        help="Maximum retained rows (default: %(default)s)",
    )
    train_sampling.add_argument(
        "--all", action="store_true", help="Retain every row (may exhaust memory)"
    )
    train_parser.add_argument(
        "--seed", type=int, default=0,
        help="Sampling, splitting, and model seed (default: %(default)s)",
    )
    train_parser.add_argument(
        "--validation-size", type=float, default=0.1,
        help="Fraction of selected rows for validation (default: %(default)s)",
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.1,
        help="Fraction of selected rows for testing (default: %(default)s)",
    )
    train_parser.add_argument(
        "--reuse-splits", type=Path,
        help="Load stored row assignments; source bytes, sample, seed, and fractions must match",
    )
    train_parser.add_argument(
        "--error-limit", type=int, default=10,
        help="Maximum error examples per evaluation; 0 disables (default: %(default)s)",
    )

    evaluate_parser = commands.add_parser(
        "evaluate", help="Evaluate a trusted model on a CSV and save a report"
    )
    evaluate_parser.add_argument("path", type=Path, help="Published headerless CSV")
    evaluate_parser.add_argument(
        "--model", type=Path, required=True, help="Trusted local model pickle only"
    )
    evaluate_parser.add_argument(
        "--report", type=Path, required=True,
        help="New JSON file; parent directory must exist",
    )
    evaluate_parser.add_argument(
        "--cohort", default="external",
        help="Evaluation cohort name in the report (default: %(default)s)",
    )
    evaluate_sampling = evaluate_parser.add_mutually_exclusive_group()
    evaluate_sampling.add_argument(
        "--sample-size", type=int, default=10_000,
        help="Maximum retained rows (default: %(default)s)",
    )
    evaluate_sampling.add_argument(
        "--all", action="store_true", help="Retain every row (may exhaust memory)"
    )
    evaluate_parser.add_argument(
        "--seed", type=int, default=0,
        help="Reproducible sampling seed (default: %(default)s)",
    )
    evaluate_parser.add_argument(
        "--error-limit", type=int, default=10,
        help="Maximum error examples; 0 disables (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            from training import train_baseline

            report = train_baseline(
                args.path,
                args.output_dir,
                sample_size=None if args.all else args.sample_size,
                seed=args.seed,
                validation_size=args.validation_size,
                test_size=args.test_size,
                reuse_splits=args.reuse_splits,
                error_limit=args.error_limit,
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "evaluate":
            from training import evaluate_file, write_report

            report = evaluate_file(
                args.path,
                args.model,
                sample_size=None if args.all else args.sample_size,
                seed=args.seed,
                cohort=args.cohort,
                error_limit=args.error_limit,
            )
            write_report(args.report, report)
            print(json.dumps(report, indent=2))
            return 0

        dataset = load_dataset(
            args.path,
            sample_size=None if args.all else args.sample_size,
            seed=args.seed,
            include_metadata=args.include_metadata,
        )
    except ImportError as error:
        parser.exit(
            2, f"error: {error}. ML commands require the training module and its "
            "dependencies; run ./bin/pip install -r requirements.txt\n"
        )
    except (DatasetError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    summary = asdict(dataset.summary)
    summary["source_counts"]["total"] = dataset.summary.source_counts.total
    summary["selected_counts"]["total"] = dataset.summary.selected_counts.total
    summary["sampled_out_rows"] = dataset.summary.sampled_out_rows
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
