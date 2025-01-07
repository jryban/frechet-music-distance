from dataclasses import dataclass
from functools import partial, reduce
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import scipy.linalg
from joblib import Memory
from numpy.typing import NDArray
from tqdm import tqdm

from .models import CLaMP2Extractor, CLaMPExtractor
from .utils import load_abc_task, load_midi_task

CAHE_MEMORY_DIR = Path.home() / ".cache" / "frechet_music_distance" / "precomputed"

memory = Memory(CAHE_MEMORY_DIR, verbose=0)
memory.reduce_size(bytes_limit="10G")


@dataclass
class FMDInfResults:
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]


class FrechetMusicDistance:

    def __init__(self, model_name: str = "clamp2", verbose: bool = True) -> None:
        self.verbose = verbose
        self.model_name = model_name
        if model_name == "clamp2":
            self.model = CLaMP2Extractor()
        elif model_name == "clamp":
            self.model = CLaMPExtractor()

        self._preprocess = partial(memory.cache(self._preprocess, ignore=["self"]), model_name=self.model_name)
        self._estimate_gaussian_parameters = memory.cache(self._estimate_gaussian_parameters, ignore=["self"])

    def score(
        self,
        reference_dataset: Union[str, Path],
        test_dataset: Union[str, Path],
        reference_ext: Optional[str] = None,
        test_ext: Optional[str] = None,
        method: str = "mle"
    ) -> float:

        reference_features= self._preprocess(reference_dataset, reference_ext)
        test_features = self._preprocess(test_dataset, test_ext)
        mean_reference, covariance_reference = self._estimate_gaussian_parameters(reference_features, method=method)
        mean_test, covariance_test = self._estimate_gaussian_parameters(test_features, method=method)

        fmd_score = self._compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)

        return fmd_score

    def score_in_memory(self, reference_dataset: list[str], test_dataset: list[str], method: str = "mle") -> float:
        reference_features = self._extract_features(reference_dataset)
        test_features = self._extract_features(test_dataset)
        mean_reference, covariance_reference = self._estimate_gaussian_parameters(reference_features, method=method)
        mean_test, covariance_test = self._estimate_gaussian_parameters(test_features, method=method)

        fmd_score = self._compute_fmd(mean_reference, mean_test, covariance_reference, covariance_test)

        return fmd_score

    def score_inf(
        self,
        reference_dataset: Union[str, Path],
        test_dataset: Union[str, Path],
        reference_ext: Optional[str] = None,
        test_ext: Optional[str] = None,
        steps: int = 25,
        min_n: int = 500,
        method: str = "mle"
    ) -> FMDInfResults:

        reference_features= self._preprocess(reference_dataset, reference_ext)
        test_features = self._preprocess(test_dataset, test_ext)
        return self._fmd_inf_pipeline(reference_features, test_features, steps, min_n, method)

    def score_inf_in_memory(
        self,
        reference_dataset: list[str],
        test_dataset: list[str],
        steps: int = 25,
        min_n: int = 500,
        method: str = "mle"
    ) -> FMDInfResults:

        reference_features = self._extract_features(reference_dataset)
        test_features = self._extract_features(test_dataset)
        return self._fmd_inf_pipeline(reference_features, test_features, steps, min_n, method)

    def _fmd_inf_pipeline(self, reference_features: NDArray, test_features: NDArray, steps: int, min_n: int, method: str) -> FMDInfResults:
        mean_reference, covariance_reference = self._estimate_gaussian_parameters(reference_features, method=method)

        score, slope, r2, points = self._compute_fmd_inf(mean_reference, covariance_reference, test_features, steps, min_n, method)

        return FMDInfResults(score, slope, r2, points)

    @staticmethod
    def clear_cache() -> None:
        memory.clear(warn=False)

    def _preprocess(
        self,
        dataset_path: Union[str, Path],
        ext: Optional[str] = None,
        model_name: str = "clamp2",
    ) -> NDArray:

        data = self._load_dataset(dataset_path, ext)
        features = self._extract_features(data)

        return features

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
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma_test.shape == sigma_ref.shape
        ), f"Training and test covariances have different dimensions, {sigma_test.shape} and {sigma_ref.shape}"

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
                raise ValueError(f"Imaginary component {m}")
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
        method: str = "mle",
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
            embds_eval = test_features[indices]

            mean_test, cov_test = self._estimate_gaussian_parameters(embds_eval, method=method)
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

    def _validate_models_to_extensions(self, model_name: str, ext: str) -> None:
        if model_name == "clamp2" and ext not in {".midi", ".mid", ".abc"}:
            raise ValueError(f"CLaMP2 model only supports .midi and .mid extensions, got {ext}")
        if model_name == "clamp" and ext not in {".abc"}:
            raise ValueError(f"CLaMP model only supports .abc extensions, got {ext}")

    def _load_dataset(self, dataset_path: Union[str, Path], file_ext: Optional[str] = None) -> Union[str, Path]:
        if file_ext is None:
            file_ext = self._get_file_ext(dataset_path)

        self._validate_models_to_extensions(self.model_name, file_ext)

        if file_ext == ".mtf" or file_ext == ".abc":
            return self._load_music_files(dataset_path, task=load_abc_task)

        elif file_ext == ".midi" or file_ext == ".mid":
            return self._load_music_files(dataset_path, task=load_midi_task)

        raise ValueError(
            f"Dataset {dataset_path} has unsupported extension {file_ext}.Supported extensions are: .midi, .mid, .mtf, .abc"
        )

    def _get_file_ext(self, dataset_path: Union[str, Path]) -> str:
        for file in Path(dataset_path).rglob("*"):
                if file.suffix in {".abc", ".midi", ".mtf", ".mid"}:
                    return file.suffix
        return None

    def _extract_features(self, data: list[str]) -> NDArray:
        features = []

        for song in tqdm(data, desc="Extracting features", disable=(not self.verbose)):
            feature = self.model.extract_feature(song)
            features.append(feature)

        return np.vstack(features)

    def _load_music_files(self, dataset_path: Union[str, Path], task: Callable):
        task_results = []
        supported_extensions = [".midi", ".mid", ".abc"]
        dataset_path = Path(dataset_path)
        file_list = reduce(
            lambda acc, arr: acc + arr,
            [[str(f) for f in dataset_path.rglob(f'**/*{file_ext}')] for file_ext in supported_extensions]
        )

        pool = ThreadPool()
        pbar = tqdm(total=len(file_list), disable=(not self.verbose), desc=f"Loading files from {dataset_path}")

        for filepath in file_list:
            res = pool.apply_async(
                task,
                args=(filepath,),
                callback=lambda *args, **kwargs: pbar.update(),
            )
            task_results.append(res)
        pool.close()
        pool.join()

        return [task.get() for task in task_results]

    def _estimate_gaussian_parameters(self, features: NDArray, method: str = "mle") -> tuple[NDArray, NDArray]:
        mean = np.mean(features, axis=0)
        covariance = np.cov(features, rowvar=False)
        return mean, covariance
