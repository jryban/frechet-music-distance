from __future__ import annotations
import logging as lg
from pathlib import Path

from ...dataset_loaders.abc_loader import ABCLoader, DatasetLoader

from ...dataset_loaders.utils import get_dataset_ext
import torch
from numpy.typing import NDArray

from ..feature_extractor import FeatureExtractor
from .clamp_model import CLaMP
from .clamp_utils import PATCH_LENGTH, MusicPatchilizer

logger = lg.getLogger(__name__)


class CLaMPExtractor(FeatureExtractor):

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        self._clamp_model_name = "sander-wood/clamp-small-1024"
        self._device = self._get_available_device()
        self._model = CLaMP.from_pretrained(self._clamp_model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

        self._patchilizer = MusicPatchilizer()
        self._softmax = torch.nn.Softmax(dim=1)

        self._patch_length = PATCH_LENGTH
        self._abc_dataset_loader = ABCLoader(verbose=verbose)


    @staticmethod
    def _get_available_device() -> torch.device:
        if torch.cuda.is_available():
            logger.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
            logger.info(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logger.info("No GPU available, using the CPU instead.")
            return torch.device("cpu")

    def _encoding_data(self, data: list[str], music_length: int) -> list[torch.Tensor]:
        """
        Encode the data into ids

        Args:
            data (list): List of strings

        Returns:
            ids_list (list): List of ids
        """
        ids_list = []
        for item in data:
            patches = self._patchilizer.encode(item, music_length=music_length, add_eos_patch=True)
            new_patches = torch.tensor(patches)
            new_patches = new_patches.reshape(-1)
            ids_list.append(new_patches)
        return ids_list

    @staticmethod
    def _abc_filter(lines: list[str]) -> str:
        """
            Filter out the metadata from the abc file

            Args:
                lines (list): List of lines in the abc file

            Returns:
                music (str): Music string
            """
        music = ""
        for line in lines:
            if line[:2] in ["A:", "B:", "C:", "D:", "F:", "G", "H:", "N:", "O:", "R:", "r:", "S:", "T:", "W:", "w:",
                            "X:", "Z:"] \
                    or line == "\n" \
                    or (line.startswith("%") and not line.startswith("%%score")):
                continue
            else:
                if "%" in line and not line.startswith("%%score"):
                    line = "%".join(line.split("%")[:-1])
                    music += line[:-1] + "\n"
                else:
                    music += line + '\n'
        return music


    def _get_features(self, ids_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Get the features from the CLaMP _model

        Args:
            ids_list (list): List of ids

        Returns:
            features_list (torch.Tensor): Tensor of features with a shape of (batch_size, hidden_size)
        """

        features_list = []
        with torch.no_grad():
            for ids in ids_list:
                ids = ids.unsqueeze(0)
                masks = torch.tensor([1] * (int(len(ids[0]) / PATCH_LENGTH))).unsqueeze(0)
                features = self._model.music_enc(ids, masks)["last_hidden_state"]
                features = self._model.avg_pooling(features, masks)
                features = self._model.music_proj(features)
                features_list.append(features[0])

        return torch.stack(features_list).to(self._device)

    @torch.no_grad()
    def _extract_feature(self, data: str) -> torch.Tensor:
        """
        Extract features from the music data

        Args:
            data (str): music data in abc format
        Returns:
            features (torch.Tensor): Extracted features

        """
        # self._abc_filter([data])
        ids = self._encoding_data([data], music_length=self._model.config.max_length)
        features = self._get_features(ids_list=ids)
        return features.detach().cpu().numpy()

    def _choose_dataset_loader(self, extension: str) -> DatasetLoader:
        if extension == ".abc":
            return self._abc_dataset_loader
        else:
            msg = f"CLAmP 2 supports .abc files but got {extension}"
            raise ValueError(msg)

    def extract_features(self, dataset_path: str | Path) -> NDArray:
        extension = get_dataset_ext(dataset_path)
        data = self._choose_dataset_loader(extension).load_dataset_async(dataset_path)

        return super()._extract_features(data)

    def extract_feature(self, filepath: str | Path) -> NDArray:
        extension = Path(filepath).suffix
        data = self._choose_dataset_loader(extension).load_file(filepath)

        return self._extract_feature(data)
