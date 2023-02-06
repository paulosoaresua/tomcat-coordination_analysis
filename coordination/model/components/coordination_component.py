from typing import Any, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import logit, sigmoid
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior
from coordination.common.utils import set_random_seed


class SigmoidGaussianCoordinationComponentParameters:

    def __init__(self, sd_uc: float, sd_c: float):
        self.sd_uc = Parameter(HalfNormalParameterPrior(np.array([sd_uc])))

    def clear_values(self):
        self.sd_uc.value = None


class SigmoidGaussianCoordinationComponentSamples:

    def __init__(self):
        self.unbounded_coordination = np.array([])
        self.coordination = np.array([])


class SigmoidGaussianCoordinationComponent:

    def __init__(self, initial_coordination: float, sd_uc: float, sd_c: float):
        self.initial_coordination = initial_coordination

        self.parameters = SigmoidGaussianCoordinationComponentParameters(sd_uc, sd_c)

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int]) -> SigmoidGaussianCoordinationComponentSamples:
        set_random_seed(seed)

        samples = SigmoidGaussianCoordinationComponentSamples()
        samples.unbounded_coordination = np.zeros((num_series, num_time_steps))
        samples.coordination = np.zeros((num_series, num_time_steps))

        # Gaussian random walk via reparametrization trick
        samples.unbounded_coordination = norm(loc=0, scale=1).rvs(
            size=(num_series, num_time_steps)) * self.parameters.sd_uc.value
        samples.unbounded_coordination[:, 0] += self.initial_coordination
        samples.unbounded_coordination = samples.unbounded_coordination.cumsum(axis=1)

        samples.coordination = sigmoid(samples.unbounded_coordination)

        return samples

    def update_pymc_model(self, time_dimension: str, unbounded_coordination_observation: Optional[Any] = None) -> Any:
        sd_uc = pm.HalfNormal(name="sd_uc", sigma=self.parameters.sd_uc.prior.sd, size=1,
                              observed=self.parameters.sd_uc.value)

        prior = pm.Normal.dist(mu=logit(self.initial_coordination), sigma=sd_uc)
        unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                       init_dist=prior,
                                                       sigma=sd_uc,
                                                       dims=[time_dimension],
                                                       observed=unbounded_coordination_observation)

        coordination = pm.Deterministic("coordination", pm.math.sigmoid(unbounded_coordination))

        return unbounded_coordination, coordination
