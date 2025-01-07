import pytest
from pathlib import Path

from frechet_music_distance.utils import load_midi_task, load_abc_task
from frechet_music_distance.fmd import FrechetMusicDistance


@pytest.fixture(scope="session", name="test_data_path")
def fixture_test_data_path() -> Path:
    return Path("tests/data").resolve(True)


@pytest.fixture(scope="session", name="midi_data_path")
def fixture_midi_data_path(test_data_path) -> Path:
    return test_data_path / "midi"


@pytest.fixture(scope="session", name="abc_data_path")
def fixture_abc_data_path(test_data_path) -> Path:
    return test_data_path / "abc"


@pytest.fixture(scope="session", name="abc_files")
def fixture_abc_files(abc_data_path) -> list[str]:
    abc_files = []
    for abc_file in abc_data_path.iterdir():
        abc_files.append(load_abc_task(abc_file))

    return abc_files

@pytest.fixture(scope="session", name="midi_files")
def fixture_midi_files(midi_data_path) -> list[str]:
    midi_files = []
    for midi_file in midi_data_path.iterdir():
        midi_files.append(load_midi_task(midi_file))

    return midi_files

@pytest.fixture(scope="function", name="fmd_clamp2")
def fixture_fmd_clamp2() -> FrechetMusicDistance:
    fmd = FrechetMusicDistance(model_name="clamp2", verbose=False)
    yield fmd
    fmd.clear_cache()
