import pytest

from frechet_music_distance.fmd import FMDInfResults
from frechet_music_distance.models import CLaMP2Extractor, CLaMPExtractor
from frechet_music_distance.utils import clear_cache


class TestFrechetMusicDistance:
    @staticmethod
    def test_basic_creation_clamp2(fmd_clamp2):
        assert fmd_clamp2 is not None
        assert fmd_clamp2.verbose is False
        assert isinstance(fmd_clamp2.feature_extractor, CLaMP2Extractor)
        clear_cache()

    @staticmethod
    def test_basic_creation_clamp(fmd_clamp):
        assert fmd_clamp is not None
        assert fmd_clamp.verbose is False
        assert isinstance(fmd_clamp.feature_extractor, CLaMPExtractor)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_files", "abc_files"])
    def test_clamp2_score(fmd_clamp2, midi_files, abc_files, input_dataset_path):
        current_dataset = locals()[input_dataset_path]
        score = fmd_clamp2.score(current_dataset, current_dataset)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_files", "abc_files"])
    def test_clamp2_score_inf(fmd_clamp2, midi_files, abc_files, input_dataset_path):
        current_dataset = locals()[input_dataset_path]
        score = fmd_clamp2.score_inf(current_dataset, current_dataset, steps=3, min_n=3)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)
        clear_cache()

    @staticmethod
    def test_clamp_score(fmd_clamp, abc_files):
        score = fmd_clamp.score(abc_files, abc_files)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)
        clear_cache()

    @staticmethod
    def test_clamp_score_inf(fmd_clamp, abc_files):
        score = fmd_clamp.score_inf(abc_files, abc_files, steps=3, min_n=3)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)
        clear_cache()


    @staticmethod
    def test_clamp_score_individual(fmd_clamp, abc_files):
        fmd = fmd_clamp
        score = fmd.score_individual(abc_files, abc_files[0])
        assert isinstance(score, float)
        assert score == pytest.approx(90, abs=1)
        clear_cache()

    @staticmethod
    def test_clamp2_score_individual_midi(fmd_clamp2, midi_files):
        fmd = fmd_clamp2
        score = fmd.score_individual(midi_files, midi_files[0])
        assert isinstance(score, float)
        assert score == pytest.approx(339, abs=1)
        clear_cache()

    @staticmethod
    def test_clamp2_score_individual_abc(fmd_clamp2, abc_files):
        fmd = fmd_clamp2
        score = fmd.score_individual(abc_files, abc_files[0])
        assert isinstance(score, float)
        assert score == pytest.approx(275, abs=1)
        clear_cache()