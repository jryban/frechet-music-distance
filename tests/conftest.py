from pathlib import Path

import pytest

from frechet_music_distance.dataloaders import ABCLoader, MIDIasMTFLoader
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


@pytest.fixture(scope="session", name="abc_files")
def fixture_abc_files(abc_data_path) -> list[str]:
    return ABCLoader().load_dataset_async(abc_data_path)

@pytest.fixture(scope="session", name="midi_files")
def fixture_midi_files(midi_data_path) -> list[str]:
    return MIDIasMTFLoader().load_dataset_async(midi_data_path)

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
