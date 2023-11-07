from __future__ import annotations
from typing import Any, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import sigmoid
from coordination.module.parametrization2 import Parameter, HalfNormalParameterPrior, \
    NormalParameterPrior
from coordination.common.utils import set_random_seed
from coordination.module.module import ModuleSamples, Module, ModuleParameters


class SigmoidGaussianCoordination(Module):
    """
    This class models a time series of continuous unbounded coordination (C) and its bounded
    version tilde{C} = sigmoid(C).
    """

    def __init__(self, mean_mean_uc0: float, sd_mean_uc0: float, sd_sd_uc: float):
        """
        Creates a coordination.

        @param mean_mean_uc0: mean of the hyper-prior of mu_uc0 (mean of the initial value of the
            unbounded coordination).
        @param sd_mean_uc0: std of the hyper-prior of mu_uc0.
        @param sd_sd_uc: std of the hyper-prior of sigma_uc (std of the Gaussian random walk of
            the unbounded coordination).
        """
        super().__init__()

        self.parameters = SigmoidGaussianCoordinationParameters(mean_mean_uc0,
                                                                sd_mean_uc0,
                                                                sd_sd_uc)

    def draw_samples(self,
                     seed: Optional[int],
                     num_series: int = None,
                     num_time_steps: int = None,
                     **kwargs) -> CoordinationSamples:
        """
        Draw coordination samples. A sample is a time series of coordination.

        @param seed: random seed for reproducibility.

        @param num_series: number of samples to generate.
        @param num_time_steps: length of each coordination series.
        @param kwargs: extra arguments to be defined by subclasses.
        @return: coordination samples. One coordination series per row.
        """
        set_random_seed(seed)

        # Gaussian random walk via re-parametrization trick
        unbounded_coordination = norm(loc=0, scale=1).rvs(
            size=(num_series, num_time_steps)) * self.parameters.sd_uc.value
        unbounded_coordination[:, 0] += self.parameters.mean_uc0.value
        unbounded_coordination = unbounded_coordination.cumsum(axis=1)

        # tilde{C} is a bounded version of coordination in the range [0,1]
        coordination = sigmoid(unbounded_coordination)

        return CoordinationSamples(unbounded_coordination=unbounded_coordination,
                                   coordination=coordination)

    def update_pymc_model(self,
                          pymc_model: pm.Model,
                          time_dimension: str = None,
                          unbounded_coordination_observed_values: Optional[Any] = None,
                          **kwargs) -> Any:
        """
        Creates the following variables in a PyMC model.
        1. Unbounded coordination at time t0 (mean_uc0).
        2. Standard deviation of the coordination Gaussian random walk (sd_uc).
        3. Unbounded coordination.
        4. Bounded coordination.

        @param pymc_model: model definition in pymc.
        @param time_dimension: name of the time dimension.
        @param unbounded_coordination_observed_values:
        @param kwargs: extra parameters to be used by child classes.
        @return: unbounded coordination, coordination, mean_uc0, sd_uc.
        """
        mean_uc0 = pm.Normal(name=self.parameters.mean_uc0.uuid,
                             mu=self.parameters.mean_uc0.prior.mean,
                             sigma=self.parameters.mean_uc0.prior.sd,
                             size=1,
                             observed=self.parameters.mean_uc0.value)
        sd_uc = pm.HalfNormal(name=self.parameters.sd_uc.uuid,
                              sigma=self.parameters.sd_uc.prior.sd,
                              size=1,
                              observed=self.parameters.sd_uc.value)

        prior = pm.Normal.dist(mu=mean_uc0, sigma=sd_uc)
        unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                       init_dist=prior,
                                                       sigma=sd_uc,
                                                       dims=[time_dimension],
                                                       observed=unbounded_coordination_observed_values)

        coordination = pm.Deterministic("coordination", pm.math.sigmoid(unbounded_coordination),
                                        dims=[time_dimension])

        return unbounded_coordination, coordination, mean_uc0, sd_uc


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class CoordinationSamples(ModuleSamples):

    def __init__(self,
                 unbounded_coordination: np.ndarray,
                 coordination: np.ndarray):
        """
        Creates an object to store coordination samples.

        @param unbounded_coordination: sampled values of an unbounded coordination variable.
            Unbounded coordination range from -Inf to +Inf.
        @param coordination: sampled coordination values in the range [0,1], or exactly 0 or 1 for
            discrete coordination.
        """
        super().__init__(coordination)

        self.unbounded_coordination = unbounded_coordination


class SigmoidGaussianCoordinationParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of the coordination module.
    """

    def __init__(self, mean_mean_uc0: float, sd_mean_uc0: float, sd_sd_uc: float):
        """
        Creates an object to store coordination parameter info.

        @param mean_mean_uc0: mean of the hyper-prior of the unbounded coordination mean at time
            t = 0.
        @param sd_mean_uc0: standard deviation of the hyper-prior of the unbounded coordination
            mean at time t = 0.
        @param sd_sd_uc: standard deviation of the hyper-prior of the standard deviation used in
            the Gaussian random walk when transitioning from one time to the next.
        """
        super().__init__()
        self.mean_uc0 = Parameter(uuid="mean_uc0",
                                  prior=NormalParameterPrior(
                                      mean=np.array([mean_mean_uc0]),
                                      sd=np.array([sd_mean_uc0]))
                                  )
        self.sd_uc = Parameter(uuid="sd_uc", prior=HalfNormalParameterPrior(np.array([sd_sd_uc])))
