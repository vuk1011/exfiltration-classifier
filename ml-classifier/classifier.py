"""Inference on precomputed request features; DNS extraction is caller-supplied.

Scores are the forest's positive-class outputs, not calibrated probabilities.
Only load trusted local pickle artifacts: unpickling can execute arbitrary code.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

if __package__:
    from .model import ModelArtifact, ModelArtifactError, load_model
    from .schema import FEATURE_NAMES
else:
    from model import ModelArtifact, ModelArtifactError, load_model
    from schema import FEATURE_NAMES


FEATURE_SCHEMA = "published_precomputed_single_request_v1"


class InvalidFeaturesError(ValueError):
    """Input does not satisfy the precomputed single-request feature contract."""


class FeatureExtractionError(ValueError):
    """A DNS query has no extractor, or its extractor failed."""


@dataclass(frozen=True)
class DNSQuery:
    """Caller-provided query metadata, preserved verbatim without DNS processing."""

    name: str
    record_type: str = "A"

    def __post_init__(self):
        for field in ("name", "record_type"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DNSQuery.{field} must be a nonempty string")


@dataclass(frozen=True)
class Prediction:
    """Immutable result; use to_dict() for an independent JSON-safe copy."""

    score: float
    threshold: float
    is_attack: bool
    features: Mapping[str, float]
    artifact_format_version: int
    feature_schema: str
    query: DNSQuery | None = None

    def __post_init__(self):
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    def to_dict(self) -> dict[str, object]:
        """Return plain JSON-safe containers, preserving query metadata verbatim."""
        return {
            "score": self.score,
            "threshold": self.threshold,
            "is_attack": self.is_attack,
            "features": dict(self.features),
            "artifact_format_version": self.artifact_format_version,
            "feature_schema": self.feature_schema,
            "query": asdict(self.query) if self.query is not None else None,
        }


# Static numeric types do not model the runtime numbers.Real registrations.
_RealValue = float | Real | np.integer | np.floating
FeatureExtractor = Callable[[DNSQuery], Mapping[str, _RealValue]]


def _threshold(value: object) -> float:
    try:
        result = (
            float(value)
            if isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            else None
        )
    except (OverflowError, ValueError) as error:
        raise ValueError("threshold must be a finite real number in [0, 1]") from error
    if result is None or not isfinite(result) or not 0 <= result <= 1:
        raise ValueError("threshold must be a finite real number in [0, 1]")
    return result


def _features(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise InvalidFeaturesError("Expected a feature mapping or DNSQuery")
    names = set(value)
    expected = set(FEATURE_NAMES)
    if names != expected:
        missing = [name for name in FEATURE_NAMES if name not in names]
        extra = [name for name in value if name not in expected]
        raise InvalidFeaturesError(
            f"Features must match FEATURE_NAMES exactly; missing={missing!r}, extra={extra!r}"
        )
    result = {}
    for name in FEATURE_NAMES:
        number = value[name]
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, Real):
            raise InvalidFeaturesError(f"Feature {name!r} must be a finite real number")
        try:
            number = float(number)
        except (OverflowError, ValueError) as error:
            raise InvalidFeaturesError(f"Feature {name!r} cannot be represented as a finite float") from error
        if not isfinite(number):
            raise InvalidFeaturesError(f"Feature {name!r} must be finite")
        if abs(number) > float(np.finfo(np.float32).max):
            raise InvalidFeaturesError(f"Feature {name!r} exceeds sklearn's float32 input range")
        result[name] = number
    return result


class DNSClassifier:
    """Reusable inference wrapper for a validated, fitted local model artifact."""

    def __init__(
        self, artifact: ModelArtifact, *, threshold: _RealValue = 0.5,
        feature_extractor: FeatureExtractor | None = None,
    ):
        self._threshold = _threshold(threshold)
        if feature_extractor is not None and not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable or None")
        contract = artifact.metadata.get("input_contract")
        if not isinstance(contract, str) or contract != FEATURE_SCHEMA:
            raise ModelArtifactError(
                f"Artifact metadata input_contract must be {FEATURE_SCHEMA!r}; got {contract!r}"
            )
        self._artifact = artifact
        self._feature_extractor = feature_extractor

    @classmethod
    def load(
        cls, path: str | Path, *, threshold: _RealValue = 0.5,
        feature_extractor: FeatureExtractor | None = None,
    ) -> "DNSClassifier":
        """Load a TRUSTED LOCAL ModelArtifact once and reuse it for inference."""
        validated_threshold = _threshold(threshold)
        return cls(
            load_model(path), threshold=validated_threshold,
            feature_extractor=feature_extractor,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def feature_schema(self) -> str:
        return FEATURE_SCHEMA

    @property
    def threshold(self) -> float:
        return self._threshold

    def classify(
        self, value: Mapping[str, _RealValue] | DNSQuery, *, feature_schema: str = FEATURE_SCHEMA,
    ) -> Prediction:
        if not isinstance(feature_schema, str) or feature_schema != FEATURE_SCHEMA:
            raise InvalidFeaturesError(f"feature_schema must be {FEATURE_SCHEMA!r}")
        query = value if isinstance(value, DNSQuery) else None
        if query is not None:
            if self._feature_extractor is None:
                raise FeatureExtractionError("DNSQuery requires a caller-supplied feature_extractor")
            try:
                value = self._feature_extractor(query)
            except Exception as error:
                raise FeatureExtractionError("feature_extractor failed for DNSQuery") from error
        features = _features(value)
        estimator = self._artifact.estimator
        try:
            classes = np.asarray(estimator.classes_)
        except (TypeError, ValueError) as error:
            raise ModelArtifactError("Estimator classes must be a binary bool array") from error
        if (
            classes.shape != (2,) or classes.dtype.kind != "b"
            or np.count_nonzero(classes) != 1
        ):
            raise ModelArtifactError("Estimator must have binary bool classes with one True class")
        positive_index = int(np.flatnonzero(classes)[0])
        output = estimator.predict_proba([list(features.values())])
        try:
            probabilities = np.asarray(output)
        except (TypeError, ValueError) as error:
            raise ModelArtifactError("predict_proba must return a numeric array with shape (1, 2)") from error
        if (
            probabilities.shape != (1, 2) or probabilities.dtype.kind not in "fiu"
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
        ):
            raise ModelArtifactError("predict_proba must return finite numeric scores in [0, 1] with shape (1, 2)")
        score = float(probabilities[0, positive_index])
        return Prediction(
            score=score, threshold=self.threshold, is_attack=score > self.threshold,
            features=features, artifact_format_version=self._artifact.format_version,
            feature_schema=FEATURE_SCHEMA, query=query,
        )
