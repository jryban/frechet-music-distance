import torch
from transformers import AutoModel, BertConfig, PreTrainedModel

from .config import CLAMP2_HIDDEN_SIZE, M3_HIDDEN_SIZE, TEXT_MODEL_NAME
from .m3_patch_encoder import M3PatchEncoder


class CLaMP2Model(PreTrainedModel):

    def __init__(
            self,
            music_config: BertConfig,
            text_model_name: str = TEXT_MODEL_NAME,
            hidden_size: int = CLAMP2_HIDDEN_SIZE
        ) -> None:

        super().__init__(music_config)

        self.text_model = AutoModel.from_pretrained(text_model_name) # Load the text model
        self.text_proj = torch.nn.Linear(self.text_model.config.hidden_size, hidden_size) # Linear layer for text projections
        torch.nn.init.normal_(self.text_proj.weight, std=0.02) # Initialize weights with normal distribution

        self.music_model = M3PatchEncoder(music_config) # Initialize the music model
        self.music_proj = torch.nn.Linear(M3_HIDDEN_SIZE, hidden_size) # Linear layer for music projections
        torch.nn.init.normal_(self.music_proj.weight, std=0.02) # Initialize weights with normal distribution

    def avg_pooling(self, input_features: torch.Tensor, input_masks: torch.Tensor) -> torch.Tensor:
        input_masks = input_masks.unsqueeze(-1).to(self.device) # add a dimension to match the feature dimension
        input_features = input_features * input_masks # apply mask to input_features
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1) # calculate average pooling

        return avg_pool

    def get_music_features(self, music_inputs: torch.Tensor, music_masks: torch.Tensor) -> torch.Tensor:
        music_features = self.music_model(music_inputs.to(self.device), music_masks.to(self.device))["last_hidden_state"]

        # Normalize features (Reduce Temporal Dimension)
        music_features = self.avg_pooling(music_features, music_masks)
        music_features = self.music_proj(music_features)

        return music_features