from transformers import PretrainedConfig, PreTrainedModel, AutoModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling


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
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()

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

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, return_loss=None,
                output_attentions=None, output_hidden_states=None, interpolate_pos_encoding=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.get_vision_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        text_outputs = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
        return BaseModelOutputWithPooling(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
        sentence_embeddings = self.mean_pooling(text_outputs, attention_mask)
        projected_outputs = self.text_projection(sentence_embeddings)
        return projected_outputs
