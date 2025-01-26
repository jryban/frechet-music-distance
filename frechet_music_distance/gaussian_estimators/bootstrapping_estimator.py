import numpy as np
from numpy.typing import NDArray

from .gaussian_estimator import GaussianEstimator
from .max_likelihood_estimator import MaxLikelihoodEstimator


class BootstrappingEstimator(GaussianEstimator):

    def __init__(self, num_samples: int = 1000) -> None:
        super().__init__()
        self._num_samples = num_samples
        self._mle = MaxLikelihoodEstimator()
        self._rng = np.random.default_rng()

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        running_mean = 0
        runing_cov = np.zeros((features.shape[1], features.shape[1]))
        for _ in range(self._num_samples):
            sample_indices = self._rng.choice(features.shape[0], size=features.shape[0], replace=True)
            bootstrap_sample = features[sample_indices]
            mean, cov = self._mle.estimate_parameters(bootstrap_sample)
            running_mean += mean / self._num_samples
            runing_cov += cov / self._num_samples

        return running_mean, runing_cov
