from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import sigmoid
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior
from coordination.common.utils import set_random_seed

from scipy.stats import beta as scipy_beta
from typing import Union


def beta(mean: Union[float, np.ndarray], var: Union[float, np.ndarray]) -> Any:
    """
    Beta distribution parameterized by a mean and a standard deviation
    """
    c = mean * (1 - mean) / var - 1
    a = mean * c
    b = (1 - mean) * c
    return scipy_beta(a, b)


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
            self.mean_uc0_name,
            self.sd_uc_name
        ]

    @property
    def mean_uc0_name(self) -> str:
        return f"mean_uc0"

    @property
    def sd_uc_name(self) -> str:
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
        mean_uc0 = pm.Normal(name=self.mean_uc0_name, mu=self.parameters.mean_uc0.prior.mean,
                             sigma=self.parameters.mean_uc0.prior.sd, size=1,
                             observed=self.parameters.mean_uc0.value)
        sd_uc = pm.HalfNormal(name=self.sd_uc_name, sigma=self.parameters.sd_uc.prior.sd, size=1,
                              observed=self.parameters.sd_uc.value)

        prior = pm.Normal.dist(mu=mean_uc0, sigma=sd_uc)
        unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                       init_dist=prior,
                                                       sigma=sd_uc,
                                                       dims=[time_dimension],
                                                       observed=unbounded_coordination_observed_values)

        coordination = pm.Deterministic("coordination", pm.math.sigmoid(unbounded_coordination))

        return unbounded_coordination, coordination, sd_uc


class BetaGaussianCoordinationComponentParameters(SigmoidGaussianCoordinationComponentParameters):

    def __init__(self, sd_mean_uc0: float, sd_sd_uc: float, sd_sd_c: float):
        super().__init__(sd_mean_uc0, sd_sd_uc)
        self.sd_c = Parameter(HalfNormalParameterPrior(np.array([sd_sd_c])))

    def clear_values(self):
        super().clear_values()
        self.sd_c.value = None


class BetaGaussianCoordinationComponentSamples(SigmoidGaussianCoordinationComponentSamples):
    pass


# For numerical stability in the Beta model
MIN_COORDINATION = 1e-16
MAX_COORDINATION = 1 - MIN_COORDINATION


class BetaGaussianCoordinationComponent:

    def __init__(self, sd_mean_uc0: float, sd_sd_uc: float, sd_sd_c: float):
        self.parameters = BetaGaussianCoordinationComponentParameters(sd_mean_uc0, sd_sd_uc, sd_sd_c)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self.mean_uc0_name,
            self.sd_uc_name,
            self.sd_c_name
        ]

    @property
    def mean_uc0_name(self) -> str:
        return f"mean_uc0"

    @property
    def sd_uc_name(self) -> str:
        return f"sd_uc"

    @property
    def sd_c_name(self) -> str:
        return f"sd_c"

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int] = None) -> BetaGaussianCoordinationComponentSamples:
        set_random_seed(seed)

        samples = BetaGaussianCoordinationComponentSamples()
        samples.unbounded_coordination = np.zeros((num_series, num_time_steps))
        samples.coordination = np.zeros((num_series, num_time_steps))

        # Gaussian random walk via reparametrization trick
        samples.unbounded_coordination = norm(loc=0, scale=1).rvs(
            size=(num_series, num_time_steps)) * self.parameters.sd_uc.value
        samples.unbounded_coordination[:, 0] += self.parameters.mean_uc0.value
        samples.unbounded_coordination = samples.unbounded_coordination.cumsum(axis=1)

        # The variance in a valid Beta distribution has to be smaller than m * (1 - m) where m is the mean of the
        # distribution
        clipped_uc = np.clip(sigmoid(samples.unbounded_coordination), MIN_COORDINATION, MAX_COORDINATION)
        clipped_vc = np.minimum(self.parameters.sd_c.value ** 2, 0.5 * clipped_uc * (1 - clipped_uc))
        samples.coordination = beta(clipped_uc, clipped_vc).rvs()

        if samples.coordination.ndim == 1:
            samples.coordination = samples.coordination[None, :]

        return samples

    def update_pymc_model(self, time_dimension: str,
                          unbounded_coordination_observed_values: Optional[Any] = None) -> Any:
        mean_uc0 = pm.Normal(name=self.mean_uc0_name, mu=self.parameters.mean_uc0.prior.mean,
                             sigma=self.parameters.mean_uc0.prior.sd, size=1,
                             observed=self.parameters.mean_uc0.value)
        sd_uc = pm.HalfNormal(name=self.sd_uc_name, sigma=self.parameters.sd_uc.prior.sd, size=1,
                              observed=self.parameters.sd_uc.value)
        sd_c = pm.HalfNormal(name=self.sd_c_name, sigma=self.parameters.sd_c.prior.sd, size=1,
                             observed=self.parameters.sd_c.value)

        prior = pm.Normal.dist(mu=mean_uc0, sigma=sd_uc)
        unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                       init_dist=prior,
                                                       sigma=sd_uc,
                                                       dims=[time_dimension],
                                                       observed=unbounded_coordination_observed_values)

        mean_coordination = pm.Deterministic("mean_coordination", pm.math.sigmoid(unbounded_coordination))
        mean_coordination_clipped = pm.Deterministic(f"mean_coordination_clipped",
                                                     pm.math.clip(mean_coordination, MIN_COORDINATION,
                                                                  MAX_COORDINATION))
        sd_c_clipped = pm.Deterministic("sd_c_clipped", pm.math.minimum(sd_c, 0.5 * mean_coordination_clipped * (
                1 - mean_coordination_clipped)))

        coordination = pm.Beta(name="coordination", mu=mean_coordination_clipped, sigma=sd_c_clipped,
                               dims=[time_dimension])

        return unbounded_coordination, coordination, sd_uc, sd_c
