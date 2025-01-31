from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from numpy.typing import NDArray
from transformers import AutoTokenizer, BertConfig

from frechet_music_distance.dataset_loaders import ABCLoader, DatasetLoader, MIDIasMTFLoader
from frechet_music_distance.dataset_loaders.utils import get_dataset_ext
from frechet_music_distance.models.feature_extractor import FeatureExtractor
from frechet_music_distance.utils import download_file

from . import config
from .clamp2_model import CLaMP2
from .m3_patchilizer import M3Patchilizer


class CLaMP2Extractor(FeatureExtractor):

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        self._accelerator = Accelerator()
        self._device = self._accelerator.device
        self._midi_dataset_loader = MIDIasMTFLoader(verbose=verbose)
        self._abc_dataset_loader = ABCLoader(verbose=verbose)

        m3_config = BertConfig(
            vocab_size=1,
            hidden_size=config.M3_HIDDEN_SIZE,
            num_hidden_layers=config.PATCH_NUM_LAYERS,
            num_attention_heads=config.M3_HIDDEN_SIZE//64,
            intermediate_size=config.M3_HIDDEN_SIZE*4,
            max_position_embeddings=config.PATCH_LENGTH
        )
        self._model = CLaMP2(m3_config, text_model_name=config.TEXT_MODEL_NAME, hidden_size=config.CLAMP2_HIDDEN_SIZE)
        self._model = self._model.to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self._patchilizer = M3Patchilizer()

        self._model.eval()

        try:
            self._checkpoint = torch.load(config.CLAMP2_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        except Exception:
            self._download_checkpoint()
            self._checkpoint = torch.load(config.CLAMP2_WEIGHTS_PATH, map_location="cpu", weights_only=True)

        self._model.load_state_dict(self._checkpoint["model"])

    def _download_checkpoint(self) -> None:
        print(f"Downloading CLaMP2 weights from: {config.CLAMP2_WEIGHTS_URL} into {config.CLAMP2_WEIGHTS_PATH}")
        download_file(config.CLAMP2_WEIGHTS_URL, config.CLAMP2_WEIGHTS_PATH, verbose=self._verbose)

    @torch.no_grad()
    def _extract_feature(self, data: str) -> NDArray:

        input_data = self._patchilizer.encode(data, add_special_patches=True)
        input_data = torch.tensor(input_data)
        max_input_length = config.PATCH_LENGTH

        segment_list = []
        for i in range(0, len(input_data), max_input_length):
            segment_list.append(input_data[i:i+max_input_length])
        segment_list[-1] = input_data[-max_input_length:]

        last_hidden_states_list = []

        for input_segment in segment_list:
            input_masks = torch.tensor([1]*input_segment.size(0))
            pad_indices = torch.ones((config.PATCH_LENGTH - input_segment.size(0), config.PATCH_SIZE)).long() * self._patchilizer.pad_token_id
            input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0)
            input_segment = torch.cat((input_segment, pad_indices), 0)
            last_hidden_states = self._model.get_music_features(music_inputs=input_segment.unsqueeze(0).to(self._device),
                                                               music_masks=input_masks.unsqueeze(0).to(self._device))
            last_hidden_states_list.append(last_hidden_states)

        full_chunk_cnt = len(input_data) // max_input_length
        remain_chunk_len = len(input_data) % max_input_length
        if remain_chunk_len == 0:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=self._device).view(-1, 1)
        else:
            feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=self._device).view(-1, 1)

        last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        last_hidden_states_list = last_hidden_states_list * feature_weights
        last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()

        return last_hidden_states_list.unsqueeze(0).detach().cpu().numpy()

    def _choose_dataset_loader(self, extension: str) -> DatasetLoader:
        if extension in (".mid", ".midi"):
            return self._midi_dataset_loader
        elif extension == ".abc":
            return self._abc_dataset_loader
        else:
            msg = f"CLAmP 2 supports .mid, .midi and .abc files but got {extension}"
            raise ValueError(msg)

    def extract_features(self, dataset_path: str | Path) -> NDArray:
        extension = get_dataset_ext(dataset_path)
        data = self._choose_dataset_loader(extension).load_dataset_async(dataset_path)

        return super()._extract_features(data)

    def extract_feature(self, filepath: str | Path) -> NDArray:
        extension = Path(filepath).suffix
        data = self._choose_dataset_loader(extension).load_file(filepath)

        return self._extract_feature(data)
