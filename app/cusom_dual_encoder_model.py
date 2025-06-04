import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

from .custom_dual_encoder_configuration import CustomDualEncoderConfig


@dataclass
class CustomDualEncoderOutput(ModelOutput):
    image_embeds: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    vision_model_last_hidden_state: Optional[torch.FloatTensor] = None
    text_model_last_hidden_state: Optional[torch.FloatTensor] = None


class CustomDualEncoderModel(PreTrainedModel):
    config_class = CustomDualEncoderConfig

    def __init__(self, config: CustomDualEncoderConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(config.vision_model_identifier).vision_model
        self.text_model = AutoModel.from_pretrained(config.text_model_identifier)

        vision_encoder_hidden_size = self.vision_model.config.hidden_size
        text_encoder_hidden_size = self.text_model.config.hidden_size
        self.text_projection = nn.Linear(text_encoder_hidden_size, vision_encoder_hidden_size)

        if config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()

    def get_image_features(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> torch.FloatTensor:
        if pixel_values is None:
            raise ValueError("pixel_values를 지정해야 합니다.")
        vision_outputs = self.vision_model(pixel_values=pixel_values, **kwargs)
        image_embeds = vision_outputs.pooler_output
        return image_embeds

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if input_ids is None:
            raise ValueError("input_ids를 지정해야 합니다.")
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else self.config.use_return_dict,
            **kwargs
        )

        pooled_output = text_outputs.pooler_output
        if pooled_output is None:
            pooled_output = text_outputs.last_hidden_state[:, 0]
        projected_text_embeds = self.text_projection(pooled_output)
        return projected_text_embeds, text_outputs.last_hidden_state

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CustomDualEncoderOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds = None
        vision_model_last_hidden_state = None
        if pixel_values is not None:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs.pooler_output if return_dict else vision_outputs[1]
            vision_model_last_hidden_state = vision_outputs.last_hidden_state if return_dict else vision_outputs[0]

        text_embeds = None
        text_model_last_hidden_state = None
        if input_ids is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = text_outputs.pooler_output if return_dict else text_outputs[1]
            if pooled_output is None and return_dict:
                 pooled_output = text_outputs.last_hidden_state[:, 0]
            elif pooled_output is None and not return_dict:
                 pooled_output = text_outputs[0][:, 0]
            text_embeds = self.text_projection(pooled_output)
            text_model_last_hidden_state = text_outputs.last_hidden_state if return_dict else text_outputs[0]

        if not return_dict:
            outputs = (image_embeds, text_embeds)
            if vision_model_last_hidden_state is not None:
                outputs = outputs + (vision_model_last_hidden_state,)
            if text_model_last_hidden_state is not None:
                outputs = outputs + (text_model_last_hidden_state,)
            return tuple(o for o in outputs if o is not None)
        return CustomDualEncoderOutput(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            vision_model_last_hidden_state=vision_model_last_hidden_state,
            text_model_last_hidden_state=text_model_last_hidden_state,
        )
