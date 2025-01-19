import numpy as np
from numpy.typing import NDArray

from .gaussian_estimator import GaussianEstimator
from .max_likelihood_estimator import MaxLikelihoodEstimator


class BootstrappingEstimator(GaussianEstimator):

    def __init__(self, num_samples: int = 1000) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.mle = MaxLikelihoodEstimator()

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        means = []
        covs = []
        for _ in range(self.num_samples):
            sample_indices = np.random.choice(features.shape[0], size=features.shape[0], replace=True)
            bootstrap_sample = features[sample_indices]
            mean, cov = self.mle.estimate_parameters(bootstrap_sample)
            means.append(mean)
            covs.append(cov)

        mean, cov = np.mean(means, axis=0), np.mean(covs, axis=0)
        return mean, cov
