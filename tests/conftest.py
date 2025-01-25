from pathlib import Path

import pytest

from frechet_music_distance.fmd import FrechetMusicDistance
from frechet_music_distance.utils import clear_cache


@pytest.fixture(scope="session", name="test_data_path")
def fixture_test_data_path() -> Path:
    return Path("tests/data").resolve(strict=True)


@pytest.fixture(scope="session", name="midi_data_path")
def fixture_midi_data_path(test_data_path) -> Path:
    return test_data_path / "midi"


@pytest.fixture(scope="session", name="abc_data_path")
def fixture_abc_data_path(test_data_path) -> Path:
    return test_data_path / "abc"


@pytest.fixture(scope="session", name="abc_song_path")
def fixture_abc_song_path(abc_data_path) -> Path:
    return abc_data_path / "example_1.abc"


@pytest.fixture(scope="session", name="midi_song_path")
def fixture_midi_song_path(midi_data_path) -> Path:
    return midi_data_path / "example_1.mid"


@pytest.fixture(scope="session", name="fmd_clamp2")
def fixture_fmd_clamp2() -> FrechetMusicDistance:
    fmd = FrechetMusicDistance(feature_extractor="clamp2", verbose=False)
    yield fmd
    clear_cache()


@pytest.fixture(scope="session", name="fmd_clamp")
def fixture_fmd_clamp() -> FrechetMusicDistance:
    fmd = FrechetMusicDistance(feature_extractor="clamp", verbose=False)
    yield fmd
    clear_cache()
