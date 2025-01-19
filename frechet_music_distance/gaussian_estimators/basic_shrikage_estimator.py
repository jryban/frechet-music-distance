from numpy.typing import NDArray
from sklearn.covariance import ShrunkCovariance

from .gaussian_estimator import GaussianEstimator


class BasicShrinkageEstimator(GaussianEstimator):

    def __init__(self, shrinkage: float = 0.1):
        super().__init__()
        self.model = ShrunkCovariance(assume_centered=False, shrinkage=shrinkage)

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        results = self.model.fit(features)

        mean = results.location_
        cov = results.covariance_
        return mean, cov
