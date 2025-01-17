import os
from typing import Tuple

import requests
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, BertConfig, PreTrainedModel

from .clamp_utils import MusicEncoder


class CLaMP(PreTrainedModel):
    """
    CLaMP model for joint text and music encoding.

    Args:
        config (:obj:`BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
        text_model_name (:obj:`str`, `optional`, defaults to :obj:`"distilroberta-base"`):
            The name of the pre-trained text model to be used for text encoding.

    Attributes:
        text_enc (:obj:`AutoModel`): The pre-trained text model used for text encoding.
        text_proj (:obj:`torch.nn.Linear`): A linear layer to project the text encoding to the hidden size of the model.
        music_enc (:obj:`MusicEncoder`): The music encoder model used for music encoding.
        music_proj (:obj:`torch.nn.Linear`): A linear layer to project the music encoding to the hidden size
        of the model.
    """

    def __init__(self, config: BertConfig, text_model_name: str = "distilroberta-base") -> None:
        super(CLaMP, self).__init__(config)
        self.text_enc = AutoModel.from_pretrained(text_model_name)
        self.text_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        torch.nn.init.normal_(self.text_proj.weight, std=0.02)

        self.music_enc = MusicEncoder(config=config)
        self.music_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        torch.nn.init.normal_(self.music_proj.weight, std=0.02)

    def forward(
        self,
        input_texts: torch.LongTensor,
        text_masks: torch.LongTensor,
        input_musics: torch.LongTensor,
        music_masks: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            input_texts (:obj:`torch.LongTensor` of shape :obj:`(batch_size, text_length)`):
                Tensor containing the integer-encoded text.
            text_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, text_length)`):
                Tensor containing the attention masks for the text.
            input_musics (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length, patch_length)`):
                Tensor containing the integer-encoded music patches.
            music_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length)`):
                Tensor containing the attention masks for the music patches.

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration
             (:class:`~transformers.BertConfig`) and inputs:
            music_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
                The music features extracted from the music encoder.
            text_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
                The text features extracted from the text encoder.
        """
        # Encode input texts
        text_features = self.text_enc(input_texts.to(self.device), attention_mask=text_masks.to(self.device))[
            "last_hidden_state"
        ]
        text_features = self.avg_pooling(text_features, text_masks)
        text_features = self.text_proj(text_features)

        # Encode input musics
        music_features = self.music_enc(input_musics, music_masks)["last_hidden_state"]
        music_features = self.avg_pooling(music_features, music_masks)
        music_features = self.music_proj(music_features)

        return music_features, text_features

    def avg_pooling(self, input_features: torch.FloatTensor, input_masks: torch.LongTensor) -> torch.FloatTensor:
        """
        Applies average pooling to the input features.

        Args:
            input_features (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length, hidden_size)`):
                Tensor containing the input features.
            input_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`):
                Tensor containing the attention masks for the input features.

        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`:
                The pooled features.
        """
        input_masks = input_masks.unsqueeze(-1).to(self.device)
        input_features = input_features * input_masks
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1)

        return avg_pool

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs) -> "CLaMP":
        """
        Instantiate a CLaMP model from a pre-trained model configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str`):
                This can be either:
                    "clamp-small-512" for the small CLaMP model with 512 max sequence length.
                    "clamp-small-1024" for the small CLaMP model with 1024 max sequence length.

        Returns:
            :class:`~transformers.CLaMP`: The CLaMP model.
        """
        model_dir = "." + pretrained_model_name_or_path

        # If the pre-trained model is not found locally, download it from Hugging Face
        if not os.path.exists(model_dir):
            # Create the model directory and download the config and pytorch model files
            print(f"Downloading CLaMP model from: {pretrained_model_name_or_path} to local machine")
            os.makedirs(model_dir)
            config_url = f"https://huggingface.co/{pretrained_model_name_or_path}/raw/main/config.json"
            model_url = f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/pytorch_model.bin"
            chunk_size = 1024 * 1024  # 1MB

            # download config file
            with requests.get(config_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(model_dir + "/config.json", "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading config") as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))

            # download pytorch model file
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(model_dir + "/pytorch_model.bin", "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading model") as pbar:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))

        # Load the model weights and configuration
        config = AutoConfig.from_pretrained(model_dir, *model_args, **kwargs)
        model = cls(config)
        state_dict = torch.load(model_dir + str("/pytorch_model.bin"), weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        return model
