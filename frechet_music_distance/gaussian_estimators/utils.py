from .shrikage_estimator import ShrinkageEstimator
from .bootstrapping_estimator import BootstrappingEstimator
from .gaussian_estimator import GaussianEstimator
from .leodit_wolf_estimator import LeoditWolfEstimator
from .max_likelihood_estimator import MaxLikelihoodEstimator
from .oas_estimator import OASEstimator


def get_estimator_by_name(name: str) -> GaussianEstimator:
    if name == "mle":
        return MaxLikelihoodEstimator()
    elif name == "bootstrap":
        return BootstrappingEstimator()
    elif name == "shrinkage":
        return ShrinkageEstimator()
    elif name == "leodit_wolf":
        return LeoditWolfEstimator()
    elif name == "oas":
        return OASEstimator()
    else:
        msg = f"Unknown estimator: {name}, valid options are: mle, bootstrap, shrinkage, leodit_wolf, oas"
        raise ValueError(msg)
