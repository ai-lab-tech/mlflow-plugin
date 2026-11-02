"""Custom configuration for the local architecture integration example."""

from transformers import BertConfig


class CustomBertConfig(BertConfig):
    model_type = "mlflow_demo_custom_bert"

    def __init__(self, logit_temperature=1.7, **kwargs):
        super().__init__(**kwargs)
        if logit_temperature <= 0:
            raise ValueError("logit_temperature must be positive")
        self.logit_temperature = float(logit_temperature)
