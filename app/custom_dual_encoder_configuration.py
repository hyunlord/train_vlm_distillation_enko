from transformers import PretrainedConfig


class CustomDualEncoderConfig(PretrainedConfig):
    model_type = "custom_dual_encoder"

    def __init__(
        self,
        vision_model_identifier="google/siglip-base-patch16-224",
        text_model_identifier="klue/bert-base",
        freeze_vision_model=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_model_identifier = vision_model_identifier
        self.text_model_identifier = text_model_identifier
        self.freeze_vision_model = freeze_vision_model
