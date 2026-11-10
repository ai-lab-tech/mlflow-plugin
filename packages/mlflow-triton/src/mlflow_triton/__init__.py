"""MLflow logging and portable Triton repositories; no deployment service required."""

from ._registration import register_model

__all__ = ["register_model"]
