from numpy.typing import NDArray
from sklearn.covariance import OAS

from .gaussian_estimator import GaussianEstimator


class OASEstimator(GaussianEstimator):

    def __init__(self) -> None:
        super().__init__()
        self._model = OAS(assume_centered=False)

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        results = self._model.fit(features)

        mean = results.location_
        cov = results.covariance_
        return mean, cov
