# Frechet Music Distance

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgements](#citation)
- [License](#license)


## Introduction
A library for calculating Frechet Music Distance (FMD). This is an official implementation of the paper [_Frechet Music Distance: A Metric For Generative Symbolic Music Evaluation_](https://www.arxiv.org/abs/2412.07948).


## Features
- Calculating FMD and FMD-Inf scores between two datasets for evaluation
- Caching extracted features and distribution parameters to speedup subsequent computations
- Support for various symbolic music representations (**MIDI** and **ABC**)
- Support for various embedding models (**CLaMP 2**)(TODO: CLaMP 1)
- Support for various methods of estimating embedding distribution parameters (TODO)
- Computation of per-song FMD to find outliers in the dataset (TODO)


## Installation

The library can be installed from from [PyPi](https://pypi.org/project/frechet-music-distance/) using pip:
```bash
pip install frechet-music-distance
```

or directly from source by cloning the repository and installing it locally:
```bash
git clone https://github.com/jryban/frechet-music-distance.git
cd frechet-music-distance
pip install -e .
```

The library was tested on Linux and MacOS, but it should Work on Windows as well.


## Usage
The library currently supports **MIDI** and **ABC** symbolic music representaions.

**Note**: When using ABC Notation please ensure that each song is located in a separate file.

### Command Line

```bash
fmd [-h] [--model {clamp2,clamp}] [--reference_ext REFERENCE_EXT] [--test_ext TEST_EXT] [--inf]
                              [--steps STEPS] [--min_n MIN_N] [--clear-cache]
                              <reference_dataset> <test_dataset>
```

#### Positional arguments:
  * `reference_dataset`:     Path to the reference dataset
  * `test_dataset`:          Path to the test dataset

#### Options:
  * `--model {clamp2,clamp}, -m {clamp2,clamp}`
                        Embedding model name
  * `--reference_ext, -r REFERENCE_EXT`
                        Music file extension in referene dataset (e.g. .midi). The program will automatically ifer this if not provided
  * `--test_ext, -t TEST_EXT`
                        Music file extension in test dataset (e.g. .midi). The program will automatically ifer this if not provided.
  * `--inf`                  Use FMD-Inf extrapolation
  * `--steps, -s STEPS`
                        Number of steps when calculating FMD-Inf
  * `--min_n, -n MIN_N`
                        Mininum sample size when calculating FMD-Inf (Must be smaller than the size of test dataset)

#### Cleanup
Additionaly the pre-computed cache can be cleared by executing:

```bash
fmd --clear-cache
```

### Python API

#### Standard FMD score
```python
from frechet_music_distance import FrechetMusicDistance

metric = FrechetMusicDistance()
score = metric.score(
    reference_dataset="<reference_dataset>",
    test_dataset="<test_dataset>"
)
```

#### FMD-Inf score
```python
from frechet_music_distance import FrechetMusicDistance

metric = FrechetMusicDistance()
result = metric.score_inf(
    reference_dataset="<reference_dataset>",
    test_dataset="<test_dataset>",
    steps=<num_steps> # default=25
    min_n=<minumum_sample_size> # default=500
)

result.score   # To get the FMD-Inf score
result.r2      # To get the R^2 of FMD-Inf linear regression
result.slope   # To get the slope of the regression
result.points  # To get the point estimates used in FMD-Inf regression

```

#### Cleanup
Additionaly the pre-computed cache can be cleared like so:

```python
from frechet_music_distance import FrechetMusicDistance

metric = FrechetMusicDistance()
metric.clear_cache()
```

## Supported Models

| Model | Name in library | Description | Creator |
| --- | --- | --- | --- |
| [CLaMP2](https://github.com/sanderwood/clamp2) | `clamp2` | CLaMP 2: Multimodal Music Information Retrieval Across 101 Languages Using Large Language Models | sanderwood |



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
* CLaMP 2: [sanderwood/clamp2](https://github.com/sanderwood/clamp2)

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE.txt) file for details.

---