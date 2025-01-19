from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from tqdm import tqdm

from .gaussian_estimators import GaussianEstimator, MaxLikelihoodEstimator
from .models import CLaMP2Extractor, FeatureExtractor


@dataclass
class FMDInfResults:
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]


class FrechetMusicDistance:

    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, gaussian_estimator: Optional[GaussianEstimator] = None, verbose: bool = True) -> None:
        self.verbose = verbose
        self.feature_extractor = feature_extractor if feature_extractor else CLaMP2Extractor(verbose=verbose)
        self.gaussian_estimator = gaussian_estimator if gaussian_estimator else MaxLikelihoodEstimator()

    def score(self, reference_dataset: list[str], test_dataset: list[str]) -> float:
        reference_features = self.feature_extractor.extract_features(reference_dataset)
        test_features = self.feature_extractor.extract_features(test_dataset)
        mean_reference, covariance_reference = self.gaussian_estimator.estimate_parameters(reference_features)
        mean_test, covariance_test = self.gaussian_estimator.estimate_parameters(test_features)

        return self._compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)

    def score_inf(
        self,
        reference_dataset: list[str],
        test_dataset: list[str],
        steps: int = 25,
        min_n: int = 500,
    ) -> FMDInfResults:

        reference_features = self.feature_extractor.extract_features(reference_dataset)
        test_features = self.feature_extractor.extract_features(test_dataset)
        mean_reference, covariance_reference = self.gaussian_estimator.estimate_parameters(reference_features)

        score, slope, r2, points = self._compute_fmd_inf(mean_reference, covariance_reference, test_features, steps, min_n)
        return FMDInfResults(score, slope, r2, points)

    def score_individual(self, reference_dataset: list[str], test_song: str) -> float:
        reference_features = self.feature_extractor.extract_features(reference_dataset)
        test_features = self.feature_extractor.extract_feature(test_song)
        mean_reference, covariance_reference = self.gaussian_estimator.estimate_parameters(reference_features)
        mean_test, covariance_test = test_features.flatten(), covariance_reference

        return self._compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)

    def _compute_fmd(
        self,
        mean_reference: NDArray,
        mean_test: NDArray,
        cov_reference: NDArray,
        cov_test: NDArray,
        eps: float = 1e-6
    ) -> float:
        mu_test = np.atleast_1d(mean_test)
        mu_ref = np.atleast_1d(mean_reference)

        sigma_test = np.atleast_2d(cov_test)
        sigma_ref = np.atleast_2d(cov_reference)

        assert (
            mu_test.shape == mu_ref.shape
        ), f"Reference and test mean vectors have different dimensions, {mu_test.shape} and {mu_ref.shape}"
        assert (
            sigma_test.shape == sigma_ref.shape
        ), f"Reference and test covariances have different dimensions, {sigma_test.shape} and {sigma_ref.shape}"

        diff = mu_test - mu_ref

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma_test.dot(sigma_ref), disp=False)
        if not np.isfinite(covmean).all():
            msg = f"FMD calculation produces singular product; adding {eps} to diagonal of cov estimates"
            if self.verbose:
                print(msg)
            offset = np.eye(sigma_test.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma_test + offset).dot(sigma_ref + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                msg = f"Imaginary component {m}"
                raise ValueError(msg)
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma_test) + np.trace(sigma_ref) - 2 * tr_covmean

    def _compute_fmd_inf(
        self,
        mean_reference: NDArray,
        cov_reference: NDArray,
        test_features: NDArray,
        steps: int = 25,
        min_n: int = 500,
    ) -> tuple[float, float, float, NDArray]:

        # Calculate maximum n
        max_n = len(test_features)

        assert min_n < max_n, f"min_n={min_n} must be smaller than number of elements in the test set: max_n={max_n}"

        # Generate list of ns to use
        ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
        results = []
        for n in tqdm(ns, desc="Calculating FMD-inf", disable=(not self.verbose)):
            # Select n feature frames randomly (with replacement)
            indices = np.random.choice(test_features.shape[0], size=n, replace=True)
            sample_test_features = test_features[indices]

            mean_test, cov_test = MaxLikelihoodEstimator().estimate_parameters(sample_test_features)
            fad_score = self._compute_fmd(mean_reference, mean_test, cov_reference, cov_test)

            # Add to results
            results.append([n, fad_score])

        # Compute FMD-inf based on linear regression of 1/n
        ys = np.array(results)
        xs = 1 / np.array(ns)
        slope, intercept = np.polyfit(xs, ys[:, 1], 1)

        # Compute R^2
        r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)

        # Since intercept is the FMD-inf, we can just return it
        return intercept, slope, r2, results
