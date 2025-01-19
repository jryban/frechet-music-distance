from numpy.typing import NDArray
from sklearn.covariance import OAS

from .gaussian_estimator import GaussianEstimator


class OASEstimator(GaussianEstimator):

    def __init__(self):
        super().__init__()
        self.model = OAS(assume_centered=False)

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        results = self.model.fit(features)

        mean = results.location_
        cov = results.covariance_
        return mean, cov
