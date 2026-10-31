"""DNS exfiltration inference from caller-supplied, precomputed features."""

# The flat source directory is mapped to dns_exfiltration_classifier at install time.
# ruff: noqa: N999

if __package__:
    from .classifier import (
        FEATURE_SCHEMA,
        DNSClassifier,
        DNSQuery,
        FeatureExtractionError,
        FeatureExtractor,
        InvalidFeaturesError,
        Prediction,
    )
    from .model import ModelArtifactError
    from .schema import FEATURE_NAMES
else:
    from classifier import (
        FEATURE_SCHEMA,
        DNSClassifier,
        DNSQuery,
        FeatureExtractionError,
        FeatureExtractor,
        InvalidFeaturesError,
        Prediction,
    )
    from model import ModelArtifactError
    from schema import FEATURE_NAMES

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "DNSClassifier",
    "DNSQuery",
    "FeatureExtractionError",
    "FeatureExtractor",
    "InvalidFeaturesError",
    "ModelArtifactError",
    "Prediction",
]
