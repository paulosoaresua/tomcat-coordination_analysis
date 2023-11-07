from __future__ import annotations
from typing import Union

import numpy as np

"""
Classes to use when defining parameter values and priors (hyper-priors).
"""


class Parameter:
    """
    This class holds parameters of a distribution.
    """

    def __init__(self, uuid: str, prior: PriorTypes, value: np.array = None):
        """
        Creates a parameter instance.

        @param uuid: unique od of the parameter.
        @param prior: prior distribution of the parameter and its parameters.
        @param value: value of the parameter.
        """

        self.uuid = uuid
        self.prior = prior
        self.value = value


class NormalParameterPrior:
    """
    Represents a normal parameter prior.
    """

    def __init__(self, mean: np.ndarray, sd: np.ndarray):
        """
        Creates a normal parameter prior.

        @param mean: mean of the normal distribution.
        @param sd: standard deviation of the normal distribution.
        """

        if (sd <= 0).any():
            raise ValueError(f"Standard deviation ({sd}) contains non-positive elements.")

        self.mean = mean
        self.sd = sd


class HalfNormalParameterPrior:
    """
    Represents a half-normal parameter prior.
    """

    def __init__(self, sd: np.ndarray):
        """
        Creates a half-normal parameter prior.

        @param sd: standard deviation of the half-normal distribution.
        """

        if (sd <= 0).any():
            raise ValueError(f"Standard deviation ({sd}) contains non-positive elements.")

        self.sd = sd


class DirichletParameterPrior:
    """
    Represents a Dirichlet parameter prior.
    """

    def __init__(self, a: np.ndarray):
        """
        Creates a Dirichlet parameter prior.

        @param a: parameter "a" of a Dirichlet distribution.
        """

        if (a <= 0).any():
            raise ValueError(f"Parameter 'a' ({a}) contains non-positive elements.")

        self.a = a


class BetaParameterPrior:
    """
    Represents a beta parameter prior.
    """

    def __init__(self, a: float, b: float):
        """
        Creates a Beta parameter prior.

        @param a: parameter "a" of a beta distribution.
        @param b: parameter "b" of a beta distribution.
        """

        if a <= 0:
            raise ValueError(f"Parameter 'a' ({a}) is non-positive.")
        if b <= 0:
            raise ValueError(f"Parameter 'a' ({a}) is non-positive.")

        self.a = a
        self.b = b


class UniformDiscreteParameterPrior:
    """
    Represents a uniform discrete parameter prior.
    """

    def __init__(self, lower: int, upper: int):
        """
        Creates a uniform discrete parameter prior.

        @param lower: smallest integer value in the sampling range.
        @param upper: largest integer value in the sampling range.
        """

        assert lower <= upper

        self.lower = lower
        self.upper = upper


PriorTypes = Union[
    NormalParameterPrior,
    HalfNormalParameterPrior,
    DirichletParameterPrior,
    BetaParameterPrior,
    UniformDiscreteParameterPrior
]
