import pytest

from frechet_music_distance import FrechetMusicDistance
from frechet_music_distance.fmd import FMDInfResults
from frechet_music_distance.models import CLaMP2Extractor, CLaMPExtractor
from frechet_music_distance.utils import clear_cache
from frechet_music_distance.gaussian_estimators.utils import get_estimator_by_name


class TestFrechetMusicDistance:
    @staticmethod
    def test_fmd_clamp2_basic_creation(base_fmd_clamp2):
        assert base_fmd_clamp2 is not None
        assert base_fmd_clamp2._verbose is False
        assert isinstance(base_fmd_clamp2._feature_extractor, CLaMP2Extractor)
        clear_cache()

    @staticmethod
    def test_basic_creation_clamp(base_fmd_clamp):
        assert base_fmd_clamp is not None
        assert base_fmd_clamp._verbose is False
        assert isinstance(base_fmd_clamp._feature_extractor, CLaMPExtractor)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_data_path", "abc_data_path"])
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score(base_fmd_clamp2, midi_data_path, abc_data_path, input_dataset_path, estimator_name):
        current_dataset = locals()[input_dataset_path]
        feature_extractor = get_estimator_by_name(estimator_name)
        base_fmd_clamp2._gaussian_estimator = feature_extractor
        score = base_fmd_clamp2.score(current_dataset, current_dataset)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_data_path", "abc_data_path"])
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score_inf(base_fmd_clamp2, midi_data_path, abc_data_path, input_dataset_path, estimator_name):
        current_dataset = locals()[input_dataset_path]
        feature_extractor = get_estimator_by_name(estimator_name)
        base_fmd_clamp2._gaussian_estimator = feature_extractor
        score = base_fmd_clamp2.score_inf(current_dataset, current_dataset, steps=3, min_n=3)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score_individual_midi(base_fmd_clamp2, midi_data_path, midi_song_path, estimator_name):
        feature_extractor = get_estimator_by_name(estimator_name)
        base_fmd_clamp2._gaussian_estimator = feature_extractor
        score = base_fmd_clamp2.score_individual(midi_data_path, midi_song_path)
        assert isinstance(score, float)
        assert score == pytest.approx(339, abs=10)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp2_score_individual_abc(base_fmd_clamp2, abc_data_path, abc_song_path, estimator_name):
        feature_extractor = get_estimator_by_name(estimator_name)
        base_fmd_clamp2._gaussian_estimator = feature_extractor
        score = base_fmd_clamp2.score_individual(abc_data_path, abc_song_path)
        assert isinstance(score, float)
        assert score == pytest.approx(275, abs=10)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp_score(base_fmd_clamp, abc_data_path, estimator_name):
        feature_extractor = get_estimator_by_name(estimator_name)
        base_fmd_clamp._gaussian_estimator = feature_extractor
        score = base_fmd_clamp.score(abc_data_path, abc_data_path)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp_score_inf(base_fmd_clamp, abc_data_path, estimator_name):
        feature_extractor = get_estimator_by_name(estimator_name)
        base_fmd_clamp._gaussian_estimator = feature_extractor
        score = base_fmd_clamp.score_inf(abc_data_path, abc_data_path, steps=3, min_n=3)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)
        clear_cache()

    @staticmethod
    @pytest.mark.parametrize("estimator_name", ["shrinkage", "mle", "leodit_wolf", "bootstrap", "oas"])
    def test_clamp_score_individual(base_fmd_clamp, abc_data_path, abc_song_path, estimator_name):
        score = base_fmd_clamp.score_individual(abc_data_path, abc_song_path)
        assert isinstance(score, float)
        assert score == pytest.approx(90, abs=10)
        clear_cache()