from transformers import PretrainedConfig, PreTrainedModel, AutoModel
import torch
import torch.nn as nn


class CombinedModelConfig(PretrainedConfig):
    model_type = "combined_sigilp2_sroberta"

    def __init__(self,
                 teacher_model_name_or_path="google/siglip2-base-patch16-224",
                 student_model_name_or_path="jhgan/ko-sroberta-multitask",
                 vision_projection_dim=1152,
                 text_projection_dim=768,
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_model_name_or_path = teacher_model_name_or_path
        self.student_model_name_or_path = student_model_name_or_path
        self.vision_projection_dim = vision_projection_dim
        self.text_projection_dim = text_projection_dim


class CombinedModel(PreTrainedModel):
    config_class = CombinedModelConfig
    base_model_prefix = "combined_sigilp2_sroberta"

    def __init__(self, config: CombinedModelConfig):
        super().__init__(config)
        self.config = config

        teacher_model = AutoModel.from_pretrained(self.config.teacher_model_name_or_path)
        self.vision_model = teacher_model.vision_model
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

        self.text_model = AutoModel.from_pretrained(self.config.student_model_name_or_path)
        self.text_projection = nn.Linear(self.config.text_projection_dim, self.config.vision_projection_dim)
        self._init_weights(self.text_projection)

        self.logit_scale = teacher_model.logit_scale
        self.logit_bias = teacher_model.logit_bias

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_vision_features(self, pixel_values=None, pixel_attention_mask=None, spatial_shapes=None, output_attentions=None, output_hidden_states=None):
        with torch.no_grad():
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            pooled_output = vision_outputs.pooler_output
        return pooled_output

    def get_text_features(self, input_ids=None, attention_mask=None, position_ids=None, output_attentions=None, output_hidden_states=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        pooled_output = text_outputs.pooler_output
        return pooled_output
