from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ..memory import MEMORY


class FeatureExtractor(ABC):

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.extract_features = MEMORY.cache(self.extract_features, ignore=["self"])

    @abstractmethod
    def extract_feature(self, data: Any) -> NDArray:
        pass

    @abstractmethod
    def extract_features(self, data: Iterable[Any]) -> NDArray:
        features = []

        for song in tqdm(data, desc="Extracting features", disable=(not self.verbose)):
            feature = self.extract_feature(song)
            features.append(feature)

        return np.vstack(features)
