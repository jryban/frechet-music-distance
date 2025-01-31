from .clamp import CLaMPExtractor
from .clamp2 import CLaMP2Extractor
from .feature_extractor import FeatureExtractor


def get_feature_extractor_by_name(name: str, verbose: bool = True) -> FeatureExtractor:
    if name == "clamp2":
        return CLaMP2Extractor(verbose=verbose)
    elif name == "clamp":
        return CLaMPExtractor(verbose=verbose)
    else:
        msg = f"Unknown feature extractor: {name}, valid options are: clamp, clamp2"
        raise ValueError(msg)
