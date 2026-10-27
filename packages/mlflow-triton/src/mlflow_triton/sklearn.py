"""Log a fitted sklearn estimator/Pipeline with a required example request."""

from ._logging import log_native_model


def log_model(sk_model, *, input_example, **kwargs):
    """Return MLflow ModelInfo with sklearn, pyfunc and Triton flavors.

    The example is the native prediction input (DataFrame/array), not a V2 HTTP
    envelope. Pass the complete fitted preprocessing Pipeline where applicable.
    """
    return log_native_model("sklearn", sk_model, input_example=input_example, **kwargs)
