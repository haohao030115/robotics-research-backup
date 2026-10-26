"""Small, dependency-light contracts shared by the research workflows.

The package deliberately contains validation and provenance only. It does not
import MuJoCo, MUSA, DDS, SONIC, or any model implementation.
"""

from .run import RunManifest, sha256_file

__all__ = [
    "CanonicalReference",
    "ReferenceValidationError",
    "load_reference",
    "RunManifest",
    "sha256_file",
]


def __getattr__(name):
    # Keep provenance-only imports usable in host inspection Python, where
    # scientific dependencies are intentionally absent.
    if name in {"CanonicalReference", "ReferenceValidationError", "load_reference"}:
        from .reference import CanonicalReference, ReferenceValidationError, load_reference

        return {"CanonicalReference": CanonicalReference,
                "ReferenceValidationError": ReferenceValidationError,
                "load_reference": load_reference}[name]
    raise AttributeError(name)
