"""Validate DNS exfiltration datasets and summarize reproducible samples."""

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
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(
            args.path,
            sample_size=None if args.all else args.sample_size,
            seed=args.seed,
            include_metadata=args.include_metadata,
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
