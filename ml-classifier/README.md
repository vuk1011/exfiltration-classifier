# DNS exfiltration dataset tools

Validate the published DNS exfiltration CSVs, load their nine single-request
features, and inspect reproducible samples. The tools do not train a model or
compute features from raw DNS queries.

## Requirements

Python 3.10 or newer. Dataset loading, the CLI, and tests use only the standard
library; no dependency installation is required for these commands.

All examples below run from `ml-classifier/`. Download the
[DNS Exfiltration Dataset](https://data.mendeley.com/datasets/c4n7fckkz3/3)
and place the CSVs in `dns_dataset/`, or supply another file path.

## Command-line usage

```sh
python3 app.py summary dns_dataset/dataset.csv
python3 app.py summary dns_dataset/dataset_modified.csv --sample-size 1000 --seed 42
python3 app.py summary --help
```

### `summary` arguments

| Argument | Default | Description |
| --- | --- | --- |
| `path` | Required | Path to a UTF-8, headerless CSV in the published 22-column format. |
| `--sample-size N` | `10000` | Maximum number of records retained; must be a positive integer. |
| `--seed N` | `0` | Integer seed for reproducible sampling. |
| `--all` | Off | Retain every record instead of sampling. Cannot be combined with `--sample-size`. |
| `--include-metadata` | Off | Retain client, domain, timestamp, and request text in loaded records. The JSON summary does not include individual records. |

The command prints a JSON summary containing:

- Resolved source path and ordered feature names.
- Full-source and selected-sample benign, attack, and total counts.
- Sampling method, requested sample size, seed, and metadata setting.
- Sampled-out and excluded row counts. No rows are silently excluded.

Save the summary using shell redirection:

```sh
python3 app.py summary dns_dataset/dataset.csv --sample-size 10000 > summary.json
```

Malformed data or invalid arguments produce an error on stderr and exit code
`2`, without a partial JSON summary.

### Sampling and memory

**Every command scans and validates the entire file**, including records not
selected for the sample. The multi-gigabyte original CSV may take several minutes.

Seeded reservoir sampling gives each record the same chance of selection. It
neither balances classes nor guarantees that a small sample contains both classes.
An unchanged file in the same order and the same seed produce the same selected
record IDs. If the requested sample exceeds the source size, all records are kept.

Sampling memory is bounded by the requested sample size. **`--all` can exhaust
memory** on large files; avoid it for the original multi-gigabyte CSV.

## Python API

Run Python from this directory so the local modules are importable:

```python
from dataset import load_dataset
from schema import FEATURE_NAMES

sample = load_dataset(
    "dns_dataset/dataset.csv",
    sample_size=10000,
    seed=0,
    include_metadata=True,
)

print(sample.summary.source_counts)
print(sample.summary.selected_counts)

for row in sample.rows:
    features = row.features  # Tuple ordered exactly as FEATURE_NAMES
    target = row.attack     # Boolean label
    source_record = (row.source, row.row_number)
    metadata = row.metadata
```

`load_dataset(path, *, sample_size=None, seed=0, include_metadata=False)` returns
a `Dataset` with `rows` and `summary`. Unlike the CLI default, omitting
`sample_size` in Python retains **all** records. Supply a bound for large files.

Each row contains the resolved source path, a one-based CSV record number, nine
numeric feature values, a boolean `attack` label, and optional metadata. Record
numbers may differ from physical line numbers when quoted fields contain newlines.
Selected rows are returned in source order. Metadata values are preserved as text
and never included in the feature vector.

For streaming processing without retaining a sample:

```python
from dataset import iter_rows

for row in iter_rows("dns_dataset/dataset.csv"):
    print(row.row_number, row.attack, row.features)
```

Iteration validates records as they are consumed. Consume the iterator fully to
validate the entire source. `load_dataset()` always consumes the full source.

Single-class files are valid inputs, including attack-only evaluation sets. Before
using a selected dataset for training, call:

```python
sample.validate_for_training()
```

This raises `DatasetError` unless the selected rows contain both classes.
Malformed source data also raises `DatasetError`; file access failures raise
`OSError`. Invalid Python argument types raise `TypeError`, and nonpositive
sample sizes raise `ValueError`.

## CSV format and validation

Files must contain all 22 published columns in this order, without a header:

```text
user_ip, domain, timestamp, attack, request,
len, subdomains_count, w_count, w_max, entropy,
w_max_ratio, w_count_ratio, digits_ratio, uppercase_ratio,
time_avg, time_stdev, size_avg, size_stdev, throughput,
unique, entropy_avg, entropy_stdev
```

Only the nine columns from `len` through `uppercase_ratio` become model inputs.
The `attack` label must be exactly `True` or `False`. Aggregate columns must be
present to preserve the schema, but their values are ignored.

Empty files, missing/extra columns, invalid labels, nonnumeric or non-finite
single-request features, malformed CSV, and invalid UTF-8 are rejected. Errors
identify the source and record or physical-line location where available.

Feature values, including entropy, are loaded unchanged—not normalized or
recomputed. The published data was filtered during aggregate-feature construction;
selecting only single-request columns does not remove that source sampling bias.
Original and modified datasets can be loaded independently.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Tests use temporary synthetic CSVs; downloading the dataset is not required.
