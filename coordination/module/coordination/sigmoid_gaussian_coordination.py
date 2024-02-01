from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import sigmoid
from coordination.common.types import TensorTypes
from coordination.common.utils import adjust_dimensions
from coordination.module.constants import (DEFAULT_NUM_TIME_STEPS,
                                           DEFAULT_UNB_COORDINATION_MEAN_PARAM,
                                           DEFAULT_UNB_COORDINATION_SD_PARAM)
from coordination.module.coordination.coordination import Coordination
from coordination.module.module import ModuleParameters, ModuleSamples
from coordination.module.parametrization2 import (HalfNormalParameterPrior,
                                                  NormalParameterPrior,
                                                  Parameter)


class SigmoidGaussianCoordination(Coordination):
    """
    This class models a time series of continuous unbounded coordination (C) and its bounded
    version tilde{C} = sigmoid(C).
    """

    def __init__(
        self,
        pymc_model: pm.Model,
        num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
        mean_mean_uc0: float = DEFAULT_UNB_COORDINATION_MEAN_PARAM,
        sd_mean_uc0: float = DEFAULT_UNB_COORDINATION_SD_PARAM,
        sd_sd_uc: float = DEFAULT_UNB_COORDINATION_SD_PARAM,
        coordination_random_variable: Optional[pm.Distribution] = None,
        mean_uc0_random_variable: Optional[pm.Distribution] = None,
        sd_uc_random_variable: Optional[pm.Distribution] = None,
        unbounded_coordination_observed_values: Optional[TensorTypes] = None,
        mean_uc0: Optional[float] = None,
        sd_uc: Optional[float] = None,
        initial_samples: Optional[np.ndarray] = None,
    ):
        """
        Creates a coordination module with an unbounded auxiliary variable.

        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_time_steps: number of time steps in the coordination scale.
        @param mean_mean_uc0: mean of the hyper-prior of mu_uc0 (mean of the initial value of the
            unbounded coordination).
        @param sd_mean_uc0: std of the hyper-prior of mu_uc0.
        @param sd_sd_uc: std of the hyper-prior of sigma_uc (std of the Gaussian random walk of
            the unbounded coordination).
        @param coordination_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param mean_uc0_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param sd_uc_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param unbounded_coordination_observed_values: observations for the unbounded coordination
            random variable. If a value is set, the variable is not latent anymore.
        @param mean_uc0: initial value of the unbounded coordination. It needs to be given for
            sampling but not for inference if it needs to be inferred. If not provided now, it can
            be set later via the module parameters variable.
        @param sd_uc: standard deviation of the unbounded coordination Gaussian random walk. It
            needs to be given for sampling but not for inference if it needs to be inferred. If
            not provided now, it can be set later via the module parameters variable.
        @param initial_samples: samples to use during a call to draw_samples. We complete with
            ancestral sampling up to the desired number of time steps.
        """
        super().__init__(
            pymc_model=pymc_model,
            parameters=SigmoidGaussianCoordinationParameters(
                module_uuid=Coordination.UUID,
                mean_mean_uc0=mean_mean_uc0,
                sd_mean_uc0=sd_mean_uc0,
                sd_sd_uc=sd_sd_uc,
            ),
            num_time_steps=num_time_steps,
            coordination_random_variable=coordination_random_variable,
            observed_values=unbounded_coordination_observed_values,
        )
        self.parameters.mean_uc0.value = mean_uc0
        self.parameters.sd_uc.value = sd_uc

        self.mean_uc0_random_variable = mean_uc0_random_variable
        self.sd_uc_random_variable = sd_uc_random_variable
        self.initial_samples = initial_samples

    def draw_samples(
        self, seed: Optional[int], num_series: int
    ) -> SigmoidGaussianCoordinationSamples:
        """
        Draw coordination samples. A sample is a time series of coordination.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if either mean_uc0 or sd_uc is None.
        @return: coordination samples. One coordination series per row.
        """
        super().draw_samples(seed, num_series)

        if self.parameters.mean_uc0.value is None:
            raise ValueError(f"Value of {self.parameters.mean_uc0.uuid} is undefined.")

        if self.parameters.sd_uc.value is None:
            raise ValueError(f"Value of {self.parameters.sd_uc.uuid} is undefined.")

        if self.initial_samples is not None:
            if self.initial_samples.shape[0] != num_series:
                raise ValueError(
                    f"The number of series {num_series} does not match the number of sampled "
                    f"series ({self.initial_samples.shape[0]}) in the provided unbounded "
                    f"coordination samples."
                )

            if self.initial_samples.shape[1] > self.num_time_steps:
                raise ValueError(
                    f"The number of time steps ({self.initial_samples.shape[1]}) in the provided "
                    f"unbounded coordination samples is larger than the requested number of time "
                    f"steps ({self.num_time_steps})."
                )

            dt = self.num_time_steps - self.initial_samples.shape[1]
            uc0 = self.initial_samples[..., -1]
        else:
            dt = self.num_time_steps
            uc0 = self.parameters.mean_uc0.value

        logging.info(
            f"Drawing {self.__class__.__name__} with {self.num_time_steps} time "
            f"steps."
        )

        if dt > 0:
            unbounded_coordination = (
                norm(loc=0, scale=1).rvs(size=(num_series, dt))
                * self.parameters.sd_uc.value
            )
            unbounded_coordination[:, 0] += uc0
            unbounded_coordination = unbounded_coordination.cumsum(axis=1)

            if self.initial_samples is not None:
                unbounded_coordination = np.concatenate(
                    [self.initial_samples, unbounded_coordination], axis=-1
                )
        else:
            unbounded_coordination = (
                np.array([]) if self.initial_samples is None else self.initial_samples
            )

        # tilde{C} is a bounded version of coordination in the range [0,1]
        coordination = sigmoid(unbounded_coordination)

        return SigmoidGaussianCoordinationSamples(
            unbounded_coordination=unbounded_coordination, coordination=coordination
        )

    def create_random_variables(self):
        """
        Creates parameters and coordination variables in a PyMC model.
        """

        with self.pymc_model:
            if self.mean_uc0_random_variable is None:
                self.mean_uc0_random_variable = pm.Normal(
                    name=self.parameters.mean_uc0.uuid,
                    mu=adjust_dimensions(
                        self.parameters.mean_uc0.prior.mean, num_rows=1
                    ),
                    sigma=adjust_dimensions(
                        self.parameters.mean_uc0.prior.sd, num_rows=1
                    ),
                    size=1,
                    observed=adjust_dimensions(
                        self.parameters.mean_uc0.value, num_rows=1
                    ),
                )
            if self.sd_uc_random_variable is None:
                self.sd_uc_random_variable = pm.HalfNormal(
                    name=self.parameters.sd_uc.uuid,
                    sigma=adjust_dimensions(self.parameters.sd_uc.prior.sd, num_rows=1),
                    size=1,
                    observed=adjust_dimensions(self.parameters.sd_uc.value, num_rows=1),
                )

            if self.coordination_random_variable is None:
                logging.info(
                    f"Fitting {self.__class__.__name__} with {self.num_time_steps} time "
                    f"steps."
                )

                # Add coordinates to the model
                if self.time_axis_name not in self.pymc_model.coords:
                    self.pymc_model.add_coord(
                        name=self.time_axis_name, values=np.arange(self.num_time_steps)
                    )

                # Create variables
                prior = pm.Normal.dist(
                    mu=self.mean_uc0_random_variable, sigma=self.sd_uc_random_variable
                )
                unbounded_coordination = pm.GaussianRandomWalk(
                    name="unbounded_coordination",
                    init_dist=prior,
                    sigma=self.sd_uc_random_variable,
                    dims=[self.time_axis_name],
                    observed=self.observed_values,
                )

                self.coordination_random_variable = pm.Deterministic(
                    name=self.uuid,
                    var=pm.math.sigmoid(unbounded_coordination),
                    dims=[self.time_axis_name],
                )


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SigmoidGaussianCoordinationParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of the coordination module.
    """

    def __init__(
        self,
        module_uuid: str,
        mean_mean_uc0: float,
        sd_mean_uc0: float,
        sd_sd_uc: float,
    ):
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
        self.mean_uc0 = Parameter(
            uuid=f"{module_uuid}_mean_uc0",
            prior=NormalParameterPrior(mean=mean_mean_uc0, sd=sd_mean_uc0),
        )
        self.sd_uc = Parameter(
            uuid=f"{module_uuid}_sd_uc",
            prior=HalfNormalParameterPrior(sd_sd_uc),
        )


class SigmoidGaussianCoordinationSamples(ModuleSamples):
    def __init__(self, unbounded_coordination: np.ndarray, coordination: np.ndarray):
        """
        Creates an object to store coordination samples.

        @param unbounded_coordination: sampled values of an unbounded coordination variable.
            Unbounded coordination range from -Inf to +Inf.
        @param coordination: sampled coordination values in the range [0,1], or exactly 0 or 1 for
            discrete coordination.
        """
        super().__init__(coordination)

        self.unbounded_coordination = unbounded_coordination
