"""Log an initialized PyTorch model through MLflow's Python prediction API."""

from ._logging import log_native_model


def log_model(pytorch_model, *, input_example, **kwargs):
    """Require an eval-mode module, MLflow-compatible example, and custom code dependencies.

    Uses pickle by default, not an implicit TorchScript/PT2 export. Preprocessing
    outside the supplied model is not captured automatically.
    """
    return log_native_model(
        "pytorch", pytorch_model, input_example=input_example, **kwargs
    )
