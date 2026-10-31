"""Small random forest and versioned, trusted-local-only pickle artifacts.

Pickle can execute code while loading. Never load an untrusted artifact; the
validation below checks compatibility, not the safety of a pickle payload.
Inputs are the nine precomputed features in schema.FEATURE_NAMES order.
"""

import pickle
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from schema import FEATURE_NAMES
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class ModelArtifactError(ValueError):
    """A serialized model artifact is malformed or incompatible."""


@dataclass
class ModelArtifact:
    estimator: RandomForestClassifier
    metadata: dict
    feature_names: tuple[str, ...] = FEATURE_NAMES
    format_version: int = 1


def create_model(seed: int = 0) -> RandomForestClassifier:
    """Return an unfitted forest; do not scale features or calibrate thresholds."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return RandomForestClassifier(
        n_estimators=5,
        criterion="entropy",
        class_weight=None,
        random_state=seed,
        max_depth=None,
    )


def _validate_artifact(artifact: ModelArtifact) -> None:
    if not isinstance(artifact, ModelArtifact):
        raise ModelArtifactError("Expected a ModelArtifact")
    if type(artifact.format_version) is not int or artifact.format_version != 1:
        raise ModelArtifactError("Unsupported model artifact format_version; expected 1")
    if not isinstance(artifact.feature_names, tuple) or artifact.feature_names != FEATURE_NAMES:
        raise ModelArtifactError("Artifact feature_names must match the expected feature order")
    if not isinstance(artifact.metadata, dict):
        raise ModelArtifactError("Artifact metadata must be a dict")
    for key in ("python_version", "numpy_version", "sklearn_version"):
        if not isinstance(artifact.metadata.get(key), str) or not artifact.metadata[key]:
            raise ModelArtifactError(f"Artifact metadata must include {key}")
    if artifact.metadata["sklearn_version"] != sklearn.__version__:
        raise ModelArtifactError(
            "Artifact sklearn_version does not match the installed version: "
            f"{artifact.metadata['sklearn_version']!r} != {sklearn.__version__!r}"
        )

    estimator = artifact.estimator
    if not isinstance(estimator, RandomForestClassifier):
        raise ModelArtifactError("Artifact estimator must be a RandomForestClassifier")
    try:
        check_is_fitted(estimator, attributes=[
            "estimators_", "classes_", "n_features_in_", "n_outputs_",
        ])
    except NotFittedError as error:
        raise ModelArtifactError("Artifact estimator must be fitted") from error
    if not estimator.estimators_:
        raise ModelArtifactError("Artifact estimator must contain fitted trees")
    if getattr(estimator, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ModelArtifactError("Estimator feature count does not match FEATURE_NAMES")
    if tuple(getattr(estimator, "feature_names_in_", FEATURE_NAMES)) != FEATURE_NAMES:
        raise ModelArtifactError("Estimator feature order does not match FEATURE_NAMES")
    classes = np.asarray(estimator.classes_)
    if (
        estimator.n_outputs_ != 1
        or classes.dtype.kind != "b"
        or classes.shape != (2,)
        or not np.array_equal(classes, [False, True])
    ):
        raise ModelArtifactError("Estimator must have binary bool classes [False, True]")


def save_model(
    path: str | Path, estimator: RandomForestClassifier, *, metadata: dict
) -> None:
    """Exclusively create an artifact; existing files are never overwritten.

    Runtime version fields are authoritative and automatically added to a copy
    of metadata. Python/NumPy versions are provenance; sklearn must match exactly
    on load because cross-version sklearn pickle loading is unsupported.
    """
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    artifact = ModelArtifact(estimator=estimator, metadata={
        **metadata,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    })
    _validate_artifact(artifact)
    with Path(path).open("xb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> ModelArtifact:
    """Load and validate a TRUSTED LOCAL pickle, never an untrusted upload."""
    with Path(path).open("rb") as stream:
        artifact = pickle.load(stream)
    _validate_artifact(artifact)
    return artifact
