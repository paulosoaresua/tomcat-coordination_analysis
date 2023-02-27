from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import logit, sigmoid
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior
from coordination.common.utils import set_random_seed


class SigmoidGaussianCoordinationComponentParameters:

    def __init__(self, sd_mean_uc0: float, sd_sd_uc: float):
        self.mean_uc0 = Parameter(NormalParameterPrior(mean=np.zeros(1), sd=np.array([sd_mean_uc0])))
        self.sd_uc = Parameter(HalfNormalParameterPrior(np.array([sd_sd_uc])))

    def clear_values(self):
        self.mean_uc0.value = None
        self.sd_uc.value = None


class SigmoidGaussianCoordinationComponentSamples:

    def __init__(self):
        self.unbounded_coordination = np.array([])
        self.coordination = np.array([])


class SigmoidGaussianCoordinationComponent:

    def __init__(self, sd_mean_uc0: float, sd_sd_uc: float):
        self.parameters = SigmoidGaussianCoordinationComponentParameters(sd_mean_uc0, sd_sd_uc)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self._mean_uc0,
            self._sd_uc
        ]

    @property
    def _mean_uc0(self) -> str:
        return f"mean_uc0"

    @property
    def _sd_uc(self) -> str:
        return f"sd_uc"

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int] = None) -> SigmoidGaussianCoordinationComponentSamples:
        set_random_seed(seed)

        samples = SigmoidGaussianCoordinationComponentSamples()
        samples.unbounded_coordination = np.zeros((num_series, num_time_steps))
        samples.coordination = np.zeros((num_series, num_time_steps))

        # Gaussian random walk via reparametrization trick
        samples.unbounded_coordination = norm(loc=0, scale=1).rvs(
            size=(num_series, num_time_steps)) * self.parameters.sd_uc.value
        samples.unbounded_coordination[:, 0] += self.parameters.mean_uc0.value
        samples.unbounded_coordination = samples.unbounded_coordination.cumsum(axis=1)

        samples.coordination = sigmoid(samples.unbounded_coordination)

        return samples

    def update_pymc_model(self, time_dimension: str,
                          unbounded_coordination_observed_values: Optional[Any] = None) -> Any:
        mean_uc0 = pm.Normal(name=self._mean_uc0, mu=self.parameters.mean_uc0.prior.mean,
                             sigma=self.parameters.mean_uc0.prior.sd, size=1,
                             observed=self.parameters.mean_uc0.value)
        sd_uc = pm.HalfNormal(name=self._sd_uc, sigma=self.parameters.sd_uc.prior.sd, size=1,
                              observed=self.parameters.sd_uc.value)

        prior = pm.Normal.dist(mu=mean_uc0, sigma=sd_uc)
        unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                       init_dist=prior,
                                                       sigma=sd_uc,
                                                       dims=[time_dimension],
                                                       observed=unbounded_coordination_observed_values)

        coordination = pm.Deterministic("coordination", pm.math.sigmoid(unbounded_coordination))

        return unbounded_coordination, coordination, sd_uc
