from typing import Any, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import logit, sigmoid
from coordination.common.distribution import beta
from coordination.common.utils import set_random_seed

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION


class BetaGaussianCoordinationComponentParameters:

    def __init__(self):
        self.sd_uc = None
        self.sd_c = None

    def reset(self):
        self.sd_uc = None
        self.sd_c = None


class BetaGaussianCoordinationComponentSamples:

    def __init__(self):
        self.unbounded_coordination = np.array([])
        self.coordination = np.array([])


class BetaGaussianCoordinationComponent:

    def __init__(self, initial_coordination: float):
        self.initial_coordination = initial_coordination

        self.parameters = BetaGaussianCoordinationComponentParameters()

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int]) -> BetaGaussianCoordinationComponentSamples:

        set_random_seed(seed)

        samples = BetaGaussianCoordinationComponentSamples()
        samples.unbounded_coordination = np.zeros((num_series, num_time_steps))
        samples.coordination = np.zeros((num_series, num_time_steps))

        samples.unbounded_coordination = norm(loc=0, scale=self.parameters.sd_uc).rvs(size=(num_series, num_time_steps))
        samples.unbounded_coordination[:, 0] += self.initial_coordination
        samples.unbounded_coordination = samples.unbounded_coordination.cumsum(axis=1)

        # A Beta distribution is only valid if var < (1 - mean) * mean. Since we use the sigmoid(unbounded coord.) as
        # the mean of a beta distribution, we have to make sure to adjust the variance properly to prevent an
        # ill-defined distribution. We also clip the mean, so it's never zero or one.
        mean_coordination = np.clip(sigmoid(samples.unbounded_coordination), MIN_COORDINATION, MAX_COORDINATION)
        var_coordination = np.minimum(self.parameters.sd_c ** 2, 0.5 * mean_coordination * (1 - mean_coordination))
        samples.coordination = beta(mean_coordination, var_coordination).rvs(size=(num_series, num_time_steps))

        return samples

    def update_pymc_model(self, time_dimension: str) -> Any:
        sd_uc = pm.HalfNormal(name="sd_uc", sigma=1, size=1, observed=self.parameters.sd_uc)
        sd_c = pm.HalfNormal(name="sd_c", sigma=1, size=1, observed=self.parameters.sd_c)

        prior = pm.Normal.dist(mu=logit(self.initial_coordination), sigma=sd_uc)
        unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                       init_dist=prior,
                                                       sigma=sd_uc,
                                                       dims=[time_dimension])

        mean_coordination = pm.Deterministic("mean_coordination", pm.math.sigmoid(unbounded_coordination))

        mean_coordination_clipped = pm.Deterministic(f"mean_coordination_clipped",
                                                     pm.math.clip(mean_coordination, MIN_COORDINATION,
                                                                  MAX_COORDINATION))
        sd_c_clipped = pm.Deterministic("sd_c_clipped", pm.math.minimum(sd_c, 0.5 * mean_coordination_clipped * (
                1 - mean_coordination_clipped)))

        coordination = pm.Beta(name="coordination", mu=mean_coordination_clipped, sigma=sd_c_clipped,
                               dims=[time_dimension])

        return unbounded_coordination, coordination
