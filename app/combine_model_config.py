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

        self.teacher_vision_model = AutoModel.from_pretrained(self.config.teacher_model_name_or_path).vision_model
        for param in self.teacher_vision_model.parameters():
            param.requires_grad = False
        self.teacher_vision_model.eval()

        self.student_text_model = AutoModel.from_pretrained(self.config.student_model_name_or_path)
        self.text_projection = nn.Linear(self.config.text_projection_dim, self.config.vision_projection_dim)

    def get_vision_features(self, pixel_values, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values를 제공해야 합니다.")
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values, **kwargs)
            vision_embeds = vision_outputs.pooler_output
        return vision_embeds

    def get_text_features(self, input_ids, attention_mask, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids와 attention_mask를 제공해야 합니다.")
        pooled_text_outputs = self.student_text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[1]
        projected_text_embeds = self.text_projection(pooled_text_outputs)
        return projected_text_embeds

    def forward(self,
                pixel_values=None,
                input_ids=None,
                attention_mask=None,
                return_dict=None,
                **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_embeds = None
        if pixel_values is not None:
            vision_embeds = self.get_vision_features(pixel_values=pixel_values, **kwargs.get('vision_kwargs', {}))

        text_embeds = None
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            text_embeds = self.get_text_features(input_ids=input_ids, attention_mask=attention_mask, **kwargs.get('text_kwargs', {}))

        if not return_dict:
            output = (vision_embeds, text_embeds)
            return tuple(x for x in output if x is not None) if any(x is not None for x in output) else None
        return {
            "vision_embeddings": vision_embeds,
            "text_embeddings": text_embeds
        }
