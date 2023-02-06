from typing import Any

import numpy as np


class Parameter:

    def __init__(self, prior: Any, value: Any = None):
        self.prior = prior
        self.value = value


class NormalParameterPrior:

    def __init__(self, mean: np.ndarray, sd: np.ndarray):
        assert (mean >= 0).all()
        assert (sd > 0).all()

        self.mean = mean
        self.sd = sd


class HalfNormalParameterPrior:

    def __init__(self, sd: np.ndarray):
        assert (sd > 0).all()

        self.sd = sd


class DirichletParameterPrior:

    def __init__(self, a: np.ndarray):
        assert (a > 0).all()

        self.a = a
