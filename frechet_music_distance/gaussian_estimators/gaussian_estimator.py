from abc import ABC, abstractmethod
from numpy.typing import NDArray
from ..memory import MEMORY

class GaussianEstimator(ABC):

    def __init__(self):
        self.estimate_parameters = MEMORY.cache(self.estimate_parameters, ignore=["self"])

    @abstractmethod
    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        pass
