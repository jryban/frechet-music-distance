import pytest

from frechet_music_distance import FrechetMusicDistance
from frechet_music_distance.fmd import FMDInfResults
from frechet_music_distance.models import CLaMP2Extractor, CLaMPExtractor
from frechet_music_distance.utils import clear_cache


class TestFrechetMusicDistance:
    @staticmethod
    def fmd():
        fmd = FrechetMusicDistance(feature_extractor="clamp2", verbose=False)
        assert fmd is not None
        assert fmd._verbose is False
        assert isinstance(fmd._feature_extractor, CLaMP2Extractor)
        clear_cache()
        del fmd

    @staticmethod
    def test_basic_creation_clamp():
        fmd = FrechetMusicDistance(feature_extractor="clamp", verbose=False)
        assert fmd is not None
        assert fmd._verbose is False
        assert isinstance(fmd._feature_extractor, CLaMPExtractor)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_data_path", "abc_data_path"])
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score(midi_data_path, abc_data_path, input_dataset_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp2", gaussian_estimator=estimator_name, verbose=False)
        current_dataset = locals()[input_dataset_path]
        score = fmd.score(current_dataset, current_dataset)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_data_path", "abc_data_path"])
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score_inf(midi_data_path, abc_data_path, input_dataset_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp2", gaussian_estimator=estimator_name, verbose=False)
        current_dataset = locals()[input_dataset_path]
        score = fmd.score_inf(current_dataset, current_dataset, steps=3, min_n=3)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score_individual_midi(midi_data_path, midi_song_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp2", gaussian_estimator=estimator_name, verbose=False)
        score = fmd.score_individual(midi_data_path, midi_song_path)
        assert isinstance(score, float)
        assert score == pytest.approx(339, abs=10)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score_individual_abc(abc_data_path, abc_song_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp2", gaussian_estimator=estimator_name, verbose=False)
        score = fmd.score_individual(abc_data_path, abc_song_path)
        assert isinstance(score, float)
        assert score == pytest.approx(275, abs=10)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp_score(abc_data_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp", gaussian_estimator=estimator_name, verbose=False)
        score = fmd.score(abc_data_path, abc_data_path)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp_score_inf(abc_data_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp", gaussian_estimator=estimator_name, verbose=False)
        score = fmd.score_inf(abc_data_path, abc_data_path, steps=3, min_n=3)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)
        clear_cache()
        del fmd

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp_score_individual(abc_data_path, abc_song_path, estimator_name):
        fmd = FrechetMusicDistance(feature_extractor="clamp", gaussian_estimator=estimator_name, verbose=False)
        score = fmd.score_individual(abc_data_path, abc_song_path)
        assert isinstance(score, float)
        assert score == pytest.approx(90, abs=10)
        clear_cache()
        del fmd
