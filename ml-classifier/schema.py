"""The published headerless CSV contract and single-request feature order."""

from dataclasses import dataclass

FEATURE_NAMES = (
    "len",
    "subdomains_count",
    "w_count",
    "w_max",
    "entropy",
    "w_max_ratio",
    "w_count_ratio",
    "digits_ratio",
    "uppercase_ratio",
)
METADATA_COLUMNS = ("user_ip", "domain", "timestamp", "attack", "request")
AGGREGATE_COLUMNS = (
    "time_avg", "time_stdev", "size_avg", "size_stdev",
    "throughput", "unique", "entropy_avg", "entropy_stdev",
)
CSV_COLUMNS = METADATA_COLUMNS + FEATURE_NAMES + AGGREGATE_COLUMNS
FEATURE_START = len(METADATA_COLUMNS)
FEATURE_STOP = FEATURE_START + len(FEATURE_NAMES)
LABEL_INDEX = CSV_COLUMNS.index("attack")


@dataclass(frozen=True)
class RequestMetadata:
    """Source values retained for analysis, never model inputs."""

    user_ip: str
    domain: str
    timestamp: str
    request: str


@dataclass(frozen=True)
class DatasetRow:
    """A feature vector with a one-based CSV record number and source path."""

    source: str
    row_number: int
    features: tuple[float, ...]
    attack: bool
    metadata: RequestMetadata | None = None


@dataclass(frozen=True)
class ClassCounts:
    benign: int = 0
    attack: int = 0

    @property
    def total(self) -> int:
        return self.benign + self.attack


@dataclass(frozen=True)
class DatasetSummary:
    source: str
    feature_names: tuple[str, ...]
    source_counts: ClassCounts
    selected_counts: ClassCounts
    sample_size: int | None
    seed: int
    include_metadata: bool
    sampling_method: str
    excluded_rows: int = 0

    @property
    def sampled_out_rows(self) -> int:
        return self.source_counts.total - self.selected_counts.total
