"""Log a supported HF pipeline or model/tokenizer component dictionary."""

from ._logging import log_native_model


def log_model(transformers_model, *, input_example, **kwargs):
    """Preserve HF assets/configuration and serve MLflow's pyfunc task interface.

    Accepts the native flavor's task, processor, model_config, code_paths and
    dependency arguments. This is unary Python serving, not a streaming LLM API.
    """
    return log_native_model(
        "transformers", transformers_model, input_example=input_example, **kwargs
    )
