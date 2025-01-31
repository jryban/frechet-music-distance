from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from ..memory import MEMORY
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class FeatureExtractor(ABC):

    def __init__(self, verbose: bool = True) -> None:
        self._verbose = verbose
        self.extract_features = MEMORY.cache(self.extract_features, ignore=["self"])

    @abstractmethod
    def _extract_feature(self, data: Any) -> NDArray:
        pass

    def _extract_features(self, data: Iterable[Any]) -> NDArray:
        features = []

        for song in tqdm(data, desc="Extracting features", disable=(not self._verbose)):
            feature = self._extract_feature(song)
            features.append(feature)

        return np.vstack(features)

    @abstractmethod
    def extract_features(self, dataset_path: str | Path) -> NDArray:
        pass

    @abstractmethod
    def extract_feature(self, filepath: str | Path) -> NDArray:
        pass
