
# Frechet Music Distance

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2412.07948v2-b31b1b.svg)](https://arxiv.org/abs/2412.07948)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Extending FMD](#extending-fmd)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Introduction
A library for calculating Frechet Music Distance (FMD). This is an official implementation of the paper [_Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation_](https://www.arxiv.org/abs/2412.07948).


## Features
- Calculating FMD and FMD-Inf scores between two datasets for evaluation
- Caching extracted features and distribution parameters to speedup subsequent computations
- Support for various symbolic music representations (**MIDI** and **ABC**)
- Support for various embedding models (**CLaMP 2**, **CLaMP 1**)
- Support for various methods of estimating embedding distribution parameters (**MLE**, **Leodit Wolf**, **Shrinkage**, **OAS**, **Bootstrap**)
- Computation of per-song FMD to find outliers in the dataset


## Installation

The library can be installed from from [PyPi](https://pypi.org/project/frechet-music-distance/) using pip:
```bash
pip install frechet-music-distance
```

**Note**: If it doesn't work try updating `pip`:
```bash
pip install --upgrade pip
```

You can also install from source by cloning the repository and installing it locally:
```bash
git clone https://github.com/jryban/frechet-music-distance.git
cd frechet-music-distance
pip install -e .
```

The library was tested on Linux and MacOS, but it should work on Windows as well.

**Note**: If you encounter `NotOpenSSLWarning` please downgrade your `urllib3` version to `1.26.6`:
```bash
pip install urllib3==1.26.6
```
You can also use a different version of Python that supports OpenSSL, by following instructions provided in [urllib3 GitHub issue](https://github.com/urllib3/urllib3/issues/3020)


## Usage
The library currently supports **MIDI** and **ABC** symbolic music representations.

**Note**: When using ABC Notation please ensure that each song is located in a separate file.

### Command Line

```bash
fmd score [-h] [--model {clamp2,clamp}] [--estimator {mle,bootstrap,oas,shrinkage,leodit_wolf}] [--inf] [--steps STEPS] [--min_n MIN_N] [--clear-cache] [reference_dataset] [test_dataset]

```

#### Positional arguments:
  * `reference_dataset`:     Path to the reference dataset
  * `test_dataset`:          Path to the test dataset

#### Options:
  * `--model {clamp2,clamp}, -m {clamp2,clamp}` Embedding model name
  * `--estimator {mle,bootstrap,oas,shrinkage,leodit_wolf}, -e {mle,bootstrap,oas,shrinkage,leodit_wolf}` Gaussian estimator for mean and covariance
  * `--inf`                  Use FMD-Inf extrapolation
  * `--steps STEPS, -s STEPS` Number of steps when calculating FMD-Inf
  * `--min_n MIN_N, -n MIN_N` Mininum sample size when calculating FMD-Inf (Must be smaller than the size of the test dataset)
  * `--clear-cache`     Clear the pre-computed cache before FMD calculation

#### Cleanup
Additionaly the pre-computed cache can be cleared by executing:

```bash
fmd clear
```

### Python API

#### Initialization
You can initialize the metric like so:

```python
from frechet_music_distance import FrechetMusicDistance

metric = FrechetMusicDistance(feature_extractor='<model_name>', gaussian_estimator='<esimator_name>', verbose=True)
```
Valid values for `<model_name>` are: `clamp2` (default), `clamp` 
Valid values for `<esimator_name>` are: `mle` (default), `bootstrap`, `shrinkage`, `leodit_wolf`, `oas`

If you want more control over feature extraction models and gaussian estimators, you can instantiate the object outside and pass it to the constructor directly like so:

```python
from frechet_music_distance import FrechetMusicDistance
from frechet_music_distance.gaussian_estimators import LeoditWolfEstimator, MaxLikelihoodEstimator, OASEstimator, BootstrappingEstimator, ShrinkageEstimator
from frechet_music_distance.models import CLaMP2Extractor, CLaMPExtractor

extractor = CLaMP2Extractor(verbose=True)
estimator = ShrinkageEstimator(shrinkage=0.1)
fmd = FrechetMusicDistance(feature_extractor=extractor, gaussian_estimator=estimator, verbose=True)

```

#### Standard FMD score
```python
score = metric.score(
    reference_dataset="<reference_dataset_path>",
    test_dataset="<test_dataset_path>"
)
```


#### FMD-Inf score
```python

result = metric.score_inf(
    reference_dataset="<reference_dataset_path>",
    test_dataset="<test_dataset_path>",
    steps=<num_steps> # default=25
    min_n=<minumum_sample_size> # default=500
)

result.score   # To get the FMD-Inf score
result.r2      # To get the R^2 of FMD-Inf linear regression
result.slope   # To get the slope of the regression
result.points  # To get the point estimates used in FMD-Inf regression

```

#### Individual (per-song) score
```python

result = metric.score_individual(
    reference_dataset="<reference_dataset_path>",
    test_song_path="<test_song_path>",
)

```

#### Cleanup
Additionaly the pre-computed cache can be cleared like so:

```python
from frechet_music_distance.utils import clear_cache

clear_cache()
```

## Extending FMD

### Embedding Model

You can add your own model as a feature extractor like so:

```python
from frechet_music_distance.models import FeatureExtractor

class MyExtractor(FeatureExtractor):

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        """<My implementation>"""
        

    @torch.no_grad()
    def _extract_feature(self, data: Any) -> NDArray:
        """<My implementation>"""


    def extract_features(self, dataset_path: str | Path) -> NDArray:
        """<My implementation of loading data>"""

        return super()._extract_features(data)


    def extract_feature(self, filepath: str | Path) -> NDArray:
        """<My implementation of loading data>"""

        return self._extract_feature(data)


```
If your model uses the same data format as CLaMP2 or CLaMP you can use `frechet_music_distance.dataset_loaders.ABCLoader` or `frechet_music_distance.dataset_loaders.MIDIasMTFLoader` for loading music data.

### Gaussian Estimator

You can add your own estimator like so:
```python
from .gaussian_estimator import GaussianEstimator
from .max_likelihood_estimator import MaxLikelihoodEstimator


class BootstrappingEstimator(GaussianEstimator):

    def __init__(self, num_samples: int = 1000) -> None:
        super().__init__()
        """<My implementation>"""

    def estimate_parameters(self, features: NDArray) -> tuple[NDArray, NDArray]:
        """<My implementation>"""

        return mean, cov
```

## Supported Embedding Models

| Model | Name in library | Description | Creator         |
| --- | --- | --- |-----------------|
| [CLaMP](https://github.com/microsoft/muzic/tree/main/clamp) | `clamp` | CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval | Microsoft Muzic |
| [CLaMP2](https://github.com/sanderwood/clamp2) | `clamp2` | CLaMP 2: Multimodal Music Information Retrieval Across 101 Languages Using Large Language Models | sanderwood      |


## Citation

If you use Frecheet Music Distance in your research, please cite the following paper:

```bibtex
@article{retkowski2024frechet,
  title={Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation},
  author={Retkowski, Jan and St{\k{e}}pniak, Jakub and Modrzejewski, Mateusz},
  journal={arXiv preprint arXiv:2412.07948},
  year={2024}
}
```

## Acknowledgements

This library uses code from the following repositories for handling the embedding models:
* CLaMP 1: [microsoft/muzic/clamp](https://github.com/microsoft/muzic/tree/main/clamp)
* CLaMP 2: [sanderwood/clamp2](https://github.com/sanderwood/clamp2)

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE.txt) file for details.

---