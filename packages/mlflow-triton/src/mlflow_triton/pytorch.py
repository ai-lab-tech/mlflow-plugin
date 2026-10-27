"""Log an eval-mode module with trained weights and custom code dependencies."""

from __future__ import annotations

from typing import Any, Literal

from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo

from ._logging import log_native_model


def log_model(
    pytorch_model: Any,
    *,
    input_example: Any,
    name: str = "model",
    signature: ModelSignature | None = None,
    triton_model_name: str | None = None,
    device: Literal["cpu", "gpu"] = "cpu",
    registered_model_name: str | None = None,
    await_registration_for: int = 300,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelInfo:
    """Log an eval-mode module with trained weights and custom code dependencies.

    ``input_example`` is required, including with an explicit signature.
    ``device`` selects the Triton target; local validation runs on CPU.
    Remaining kwargs go to the native MLflow save_model API. See the core
    README's API reference for framework-specific options and limitations.
    """
    return log_native_model(
        "pytorch",
        pytorch_model,
        input_example=input_example,
        name=name,
        signature=signature,
        triton_model_name=triton_model_name,
        device=device,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        run_id=run_id,
        metadata=metadata,
        tags=tags,
        params=params,
        **kwargs,
    )
