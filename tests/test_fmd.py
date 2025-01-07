import pytest
from frechet_music_distance.models import CLaMP2Extractor
from frechet_music_distance.fmd import FMDInfResults

class TestFrechetMusicDistance:
    @staticmethod
    def test_basic_creation_clamp2(fmd_clamp2):
        assert fmd_clamp2 is not None
        assert fmd_clamp2.model_name == "clamp2"
        assert fmd_clamp2.verbose is False
        assert isinstance(fmd_clamp2.model, CLaMP2Extractor)

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_data_path", "abc_data_path"])
    def test_clamp2_score(fmd_clamp2, midi_data_path, abc_data_path, input_dataset_path):
        current_input = locals()[input_dataset_path]
        score = fmd_clamp2.score(current_input, current_input)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_data_path", "abc_data_path"])
    def test_clamp2_score_inf(fmd_clamp2, midi_data_path, abc_data_path, input_dataset_path):
        current_input = locals()[input_dataset_path]
        score = fmd_clamp2.score_inf(current_input, current_input, steps=10, min_n=2)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_files", "abc_files"])
    def test_clamp2_score_in_memory(fmd_clamp2, midi_files, abc_files, input_dataset_path):
        current_dataset = locals()[input_dataset_path]
        score = fmd_clamp2.score_in_memory(current_dataset, current_dataset)
        assert isinstance(score, float)
        assert score == pytest.approx(0, abs=0.1)

    @staticmethod
    @pytest.mark.parametrize("input_dataset_path", ["midi_files", "abc_files"])
    def test_clamp2_score_inf_in_memory(fmd_clamp2, midi_files, abc_files, input_dataset_path):
        current_dataset = locals()[input_dataset_path]
        score = fmd_clamp2.score_inf_in_memory(current_dataset, current_dataset, steps=10, min_n=2)
        assert isinstance(score, FMDInfResults)
        assert isinstance(score.score, float)
        assert isinstance(score.r2, float)
        assert isinstance(score.slope, float)
        assert isinstance(score.points, list)