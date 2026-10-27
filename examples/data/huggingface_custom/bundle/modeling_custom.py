"""Custom inference behavior using the saved, unchanged BERT parameters."""

from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_custom import CustomBertConfig


class CustomBertForSequenceClassification(BertForSequenceClassification):
    config_class = CustomBertConfig

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if labels is not None:
            raise ValueError("This integration example is inference-only")
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = output.logits / self.config.logit_temperature
        if return_dict is False or (
            return_dict is None and not self.config.use_return_dict
        ):
            return (logits,) + output.to_tuple()[1:]
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
