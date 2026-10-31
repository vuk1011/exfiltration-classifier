# DNS exfiltration classifier

Classify a single DNS request using a trained random forest. The library accepts
**nine named features**, or a `DNSQuery` object with a caller-supplied feature
extractor. It also provides tools for CSV validation, sampling, training, and
evaluation. DNS parsing and feature extraction are not bundled.

## Requirements

Dataset loading and `summary` require Python 3.10 or newer and use only the
standard library; importing `app` does not import the ML dependencies. Training,
evaluation, and the full test suite require Python **3.12 or newer** with the
scikit-learn and NumPy versions pinned in `requirements.txt`.

All examples below run from `ml-classifier/`. Use the existing local environment:

```sh
./bin/pip install -r requirements.txt
```

If the local environment is missing or incompatible with your OS, recreate it
with `python3 -m venv .` and install the dependencies above. The explicit
`./bin/python` commands below use that interpreter without activation.

For training and CSV inspection, download the
[DNS Exfiltration Dataset](https://data.mendeley.com/datasets/c4n7fckkz3/3)
and place the CSVs in `dns_dataset/`, or supply another file path. The dataset is
**not required for inference**.

## Install and deploy

Install from this source directory into your application's Python environment:

```sh
python3 -m pip install .
```

To deploy without copying the source tree or dataset, build a wheel from
`ml-classifier/`:

```sh
./bin/python -m pip install setuptools wheel
./bin/python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

Copy these two files to the receiving machine:

- `dist/dns_exfiltration_classifier-0.1.0-py3-none-any.whl`
- `runs/baseline/model.pkl`

Then install the wheel in the receiving application's environment:

```sh
python3 -m pip install dns_exfiltration_classifier-0.1.0-py3-none-any.whl
```

Pip installs the pinned runtime dependencies. Use Python 3.12 or newer. The model
loader requires the same scikit-learn version as the saved artifact; matching the
pinned NumPy version is also recommended. Platform-specific dependency wheels
must be available for the receiving machine.

The existing baseline pickle is supported without conversion or retraining.
The wheel contains library code, not datasets, model artifacts, tests, or the
source machine's virtual environment. Neither `splits.json` nor `report.json` is
needed to classify a request. Load only model files you trust: pickle loading can
execute arbitrary code.

## Inference API

### Named features

```python
import json

from dns_exfiltration_classifier import DNSClassifier

classifier = DNSClassifier.load("model.pkl")

features = {
    "len": 6,
    "subdomains_count": 1,
    "w_count": 2,
    "w_max": 3,
    "entropy": 2.584962500721156,
    "w_max_ratio": 0.5,
    "w_count_ratio": 1 / 3,
    "digits_ratio": 0.0,
    "uppercase_ratio": 0.0,
}
result = classifier.classify(features)
print(result.is_attack, result.score)
print(json.dumps(result.to_dict()))
```

Load the model once and reuse the classifier. A mapping must contain exactly the
nine feature names; insertion order does not matter. Values must be finite real
numbers, not booleans or numeric strings. No scaling, normalization, or training
occurs during classification.

### DNS query objects

Python's standard library has no DNS query message type. This library supplies a
small immutable `DNSQuery(name, record_type="A")` dataclass. It preserves name case
and trailing dots verbatim; it does not parse wire packets, resolve names, strip
domains, or validate DNS syntax beyond nonempty strings.

Supply your independently implemented extractor when loading the model:

```python
from dns_exfiltration_classifier import DNSClassifier, DNSQuery

# extract_features is your implementation: DNSQuery -> mapping of nine features.
classifier = DNSClassifier.load("model.pkl", feature_extractor=extract_features)
query = DNSQuery(name="encoded.example.test.", record_type="TXT")
result = classifier.classify(query)
```

The callback receives the query object once. Its returned mapping is validated
and classified through the same path as direct features. You may pass a feature
mapping to this classifier as well; doing so bypasses the callback. Convert a
question from your DNS parser into `DNSQuery` at your application's boundary.

**An extractor is required for query input.** The library does not guess entropy,
word matching, or base-domain removal. Your extractor must match the training
semantics for all nine features, including case, punctuation, dictionary behavior,
and label removal. Matching feature names alone is not proof of compatibility.

### Parameters and results

`DNSClassifier.load(path, *, threshold=0.5, feature_extractor=None)` loads a trusted
artifact and checks its feature schema. `classify(value, *, feature_schema=FEATURE_SCHEMA)`
accepts a feature mapping or `DNSQuery`. The public `FEATURE_SCHEMA` identifier is
`published_precomputed_single_request_v1`; mismatches are rejected. This identifier
checks the declared contract, not the correctness of your extraction implementation.

`Prediction` contains:

- `score`: the random forest's score for the attack class, **not a calibrated risk
  probability or confidence guarantee**.
- `threshold` and `is_attack`: attack is strictly `score > threshold`. The default
  `0.5` agrees with the forest's normal decision rule; a score equal to the
  threshold is benign. Custom thresholds must be finite and in `[0, 1]`, and are
  policy choices, not automatically validated settings.
- `features`: an immutable mapping in canonical feature order.
- `feature_schema` and `artifact_format_version`: contract/serialization versions,
  not a unique identifier for a particular training run.
- `query`: the original `DNSQuery` when supplied, otherwise `None`.

Use `result.to_dict()` for an independent, JSON-safe copy. The classifier exposes
`feature_names`, `feature_schema`, and `threshold` properties.

### Invalid inputs

- `InvalidFeaturesError`: missing/extra features, invalid values, or a mismatched
  declared feature schema.
- `FeatureExtractionError`: query input without an extractor, or an extractor
  exception. The original exception is available as `__cause__`.
- `ModelArtifactError`: an incompatible artifact or invalid estimator output.

These exceptions are exported from `dns_exfiltration_classifier`. Invalid input
never becomes a benign prediction: the caller must decide how to handle errors.
File-access failures and corrupt pickle errors also propagate; compatibility
checks do not make untrusted artifacts safe.

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

## Training and evaluation

```sh
./bin/python app.py train dns_dataset/dataset.csv --output-dir runs/baseline
./bin/python app.py train dns_dataset/dataset.csv --output-dir runs/reused \
  --reuse-splits runs/baseline/splits.json
./bin/python app.py evaluate dns_dataset/dataset_modified.csv \
  --model runs/baseline/model.pkl --report modified-report.json --cohort modified
./bin/python app.py train --help
./bin/python app.py evaluate --help
```

Both commands print the returned report as JSON on stdout. Training creates
`model.pkl`, `splits.json`, and `report.json` inside `--output-dir`. **The output
directory must not already exist**; existing artifacts are never overwritten.
Evaluation saves JSON to the required `--report` path before printing it. That
file must not exist and its parent directory must already exist.

### Arguments

| Argument | `train` default | `evaluate` default | Description |
| --- | --- | --- | --- |
| `path` | Required | Required | Published headerless CSV, not raw queries. |
| `--output-dir PATH` | Required | Not available | New directory for training artifacts. |
| `--model PATH` | Not available | Required | Trusted local model pickle. |
| `--report PATH` | Not available | Required | New JSON report file in an existing directory. |
| `--sample-size N` | `10000` | `10000` | Positive maximum number of retained rows. |
| `--all` | Off | Off | Retain every row; mutually exclusive with `--sample-size`. |
| `--seed N` | `0` | `0` | Reproducible sampling and, for training, splitting/model seed (`0` through `2**32 - 1`). |
| `--validation-size F` | `0.1` | Not available | Fraction of selected rows for validation. |
| `--test-size F` | `0.1` | Not available | Fraction of selected rows for testing. |
| `--reuse-splits PATH` | None | Not available | Load stored row assignments from a previous `splits.json`. |
| `--cohort NAME` | Not available | `external` | Name identifying the evaluation cohort. |
| `--error-limit N` | `10` | `10` | Nonnegative bound on selected misclassified examples per evaluation. |

Validation and test fractions must leave a nonempty training fraction and enough
rows in each class for stratification. The default bounded sample of **10,000**
rows may contain too few minority-class examples. Increase `--sample-size` if
needed; a larger sample still does not guarantee sufficient minority counts.

`--reuse-splits` requires matching source path, size, modification time, sample
size, seed, and split fractions. It loads the stored row assignments rather than
creating new ones. Use a different, nonexistent output directory for the new run.
Split manifests use format version 2; older manifests must be regenerated.

### Baseline and interpretation

The paper-style baseline is a random forest with **5 trees**, **entropy** splits,
and **no feature scaling**. By default the selected rows are divided into an
**80/10/10 stratified random-row train/validation/test split**. Predictions use the
model's fixed default `predict()` decision rule; there is **no threshold tuning**.

Reports include class counts, TP/FP/TN/FN, precision, recall, F1, false-positive
rate, training and prediction time, model size, global feature importance, and a
bounded list of misclassified examples. Prediction timing excludes CSV loading and
feature extraction. Global feature importance is not a per-query explanation.

Leakage audits report exact raw-request and nine-feature-vector overlap across
splits; they **do not eliminate leakage**.
Random-row results are not evidence of generalization to unseen clients, domains,
or time periods. External evaluation reports include overlap with the artifact's
stored selected row IDs when evaluating the same source; do not treat overlapping
rows as an independent holdout.

The modified dataset is attack-only. Its report can measure attack recall, but
**precision, F1, and false-positive rate are JSON `null`**, not zero or evidence
of perfect performance. A cohort containing benign examples is needed to measure
false alarms.

Training and evaluation scan and validate the **entire source**, even when
sampling. Source checks use only the resolved path, file size, and modification
time; **no content hash is generated and there is no extra content-reading pass**.
These checks do not prove byte-for-byte identity: edits preserving size and
modification time at the same path cannot be detected. Full-source parsing can
still take several minutes on multi-gigabyte inputs. `--all` removes the sample
memory bound and can exhaust memory.

### Artifact safety

**Load only trusted pickle files.** Unpickling a model can execute arbitrary code;
artifact compatibility checks do not make an untrusted pickle safe.

Workflow reports include selected misclassified examples with **raw query
metadata**, potentially including client addresses, domains, timestamps, and
request text. Treat reports and model/split artifacts as potentially sensitive;
review them before sharing or committing. `--error-limit 0` disables selected
error examples, but is not a general artifact-redaction guarantee.

Malformed inputs, invalid workflow settings, missing dependencies, and I/O
failures exit with code `2`, reporting an actionable error on stderr without
partial report JSON on stdout. Unexpected programming errors are not hidden.

## Python API

For dataset tooling, the existing flat imports work when Python runs from this
directory. When installed, use `dns_exfiltration_classifier.dataset` and
`dns_exfiltration_classifier.schema` instead:

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
./bin/python -m unittest discover -s tests -v
```

Tests use temporary synthetic CSVs; downloading the dataset is not required.
Read-only checks of the existing baseline are skipped when its pickle is absent.
Wheel-installation tests require `setuptools` and `wheel`; they install into a
temporary directory and check imports outside the source checkout without network
access. CLI training/evaluation dispatch tests mock the flat `training` APIs, so
those tests do not need real training artifacts:

```sh
./bin/python -m unittest discover -s tests -p 'test_training_app.py' -v
./bin/python -m unittest discover -s tests -p 'test_app.py' -v
```
