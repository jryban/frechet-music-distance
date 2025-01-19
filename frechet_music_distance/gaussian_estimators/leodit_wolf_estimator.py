from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

from .gaussian_estimator import GaussianEstimator


class LeoditWolfEstimator(GaussianEstimator):

    def __init__(self, block_size: int = 1000):
        super().__init__()
        self.model = LedoitWolf(assume_centered=False, block_size=block_size)

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        results = self.model.fit(features)

        mean = results.location_
        cov = results.covariance_
        return mean, cov
