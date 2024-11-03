from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pymc as pm
from scipy.stats import norm
import pytensor.tensor as ptt

from coordination.common.functions import sigmoid
from coordination.common.types import TensorTypes
from coordination.common.utils import adjust_dimensions
from coordination.module.constants import (DEFAULT_NUM_TIME_STEPS,
                                           DEFAULT_UNB_COORDINATION_MEAN_PARAM,
                                           DEFAULT_UNB_COORDINATION_SD_PARAM)
from coordination.module.coordination.coordination import Coordination
from coordination.module.module import ModuleParameters, ModuleSamples
from coordination.module.parametrization import (HalfNormalParameterPrior,
                                                 NormalParameterPrior,
                                                 Parameter)


class SigmoidGaussianCoordination3D(Coordination):
    """
    This class models a time series of continuous unbounded coordination (U) and its bounded
    version C = sigmoid(U) / sigmoid(U).sum()[None, :] normalized such that the values in each
    axis sum up to 1. Coordination here is (3, T) having the axes representing: individualism,
    coordination and common cause. Individualism controls how much each person is in sync with
    itself across time. Coordination controls how much each person is in sync with their partners
    across time. Finally, common cause controls how much each person is in sync with an external
    unknown source (e.g., the environment).
    """

    def __init__(
            self,
            pymc_model: pm.Model,
            num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
            mean_mean_uc0: float = DEFAULT_UNB_COORDINATION_MEAN_PARAM,
            sd_mean_uc0: float = DEFAULT_UNB_COORDINATION_SD_PARAM,
            sd_sd_uc: float = DEFAULT_UNB_COORDINATION_SD_PARAM,
            coordination_random_variable: Optional[pm.Distribution] = None,
            mean_uc0_individualism_random_variable: Optional[pm.Distribution] = None,
            sd_uc_individualism_random_variable: Optional[pm.Distribution] = None,
            mean_uc0_coordination_random_variable: Optional[pm.Distribution] = None,
            sd_uc_coordination_random_variable: Optional[pm.Distribution] = None,
            mean_uc0_common_cause_random_variable: Optional[pm.Distribution] = None,
            sd_uc_common_cause_random_variable: Optional[pm.Distribution] = None,
            unbounded_individualism_observed_values: Optional[TensorTypes] = None,
            unbounded_coordination_observed_values: Optional[TensorTypes] = None,
            unbounded_common_cause_observed_values: Optional[TensorTypes] = None,
            mean_uc0_individualism: Optional[float] = None,
            sd_uc_individualism: Optional[float] = None,
            mean_uc0_coordination: Optional[float] = None,
            sd_uc_coordination: Optional[float] = None,
            mean_uc0_common_cause: Optional[float] = None,
            sd_uc_common_cause: Optional[float] = None,
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
            create_random_variables. If not set, it will be created in such a call. This variable's
            shape is (3, T).
        @param mean_uc0_individualism_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param sd_uc_individualism_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param mean_uc0_coordination_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param sd_uc_coordination_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param mean_uc0_common_cause_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param sd_uc_common_cause_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param unbounded_individualism_observed_values: observations for the unbounded
            individualism random variable. If a value is set, the variable is not latent anymore.
        @param unbounded_coordination_observed_values: observations for the unbounded coordination
            random variable. If a value is set, the variable is not latent anymore.
        @param unbounded_common_cause_observed_values: observations for the unbounded common cause
            random variable. If a value is set, the variable is not latent anymore.
        @param mean_uc0_individualism: initial value of the unbounded individualism. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        @param sd_uc_individualism: standard deviation of the unbounded individualism Gaussian
            random walk. It needs to be given for sampling but not for inference if it needs to be
            inferred. If not provided now, it can be set later via the module parameters variable.
        @param mean_uc0_coordination: initial value of the unbounded coordination. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        @param sd_uc_coordination: standard deviation of the unbounded coordination Gaussian
            random walk. It needs to be given for sampling but not for inference if it needs to be
            inferred. If not provided now, it can be set later via the module parameters variable.
        @param mean_uc0_common_cause: initial value of the unbounded common cause. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        @param sd_uc_common_cause: standard deviation of the unbounded common cause Gaussian
            random walk. It needs to be given for sampling but not for inference if it needs to be
            inferred. If not provided now, it can be set later via the module parameters variable.
        @param initial_samples: samples to use during a call to draw_samples. We complete with
            ancestral sampling up to the desired number of time steps. This variable has shape
            (3, T).
        """
        super().__init__(
            pymc_model=pymc_model,
            parameters=SigmoidGaussianCoordinationParameters3D(
                module_uuid=Coordination.UUID,
                mean_mean_uc0=mean_mean_uc0,
                sd_mean_uc0=sd_mean_uc0,
                sd_sd_uc=sd_sd_uc,
            ),
            num_time_steps=num_time_steps,
            coordination_random_variable=coordination_random_variable,
            observed_values=None,
        )
        self.parameters.mean_uc0_individualism.value = mean_uc0_individualism
        self.parameters.sd_uc_individualism.value = sd_uc_individualism
        self.parameters.mean_uc0_coordination.value = mean_uc0_coordination
        self.parameters.sd_uc_coordination.value = sd_uc_coordination
        self.parameters.mean_uc0_common_cause.value = mean_uc0_common_cause
        self.parameters.sd_uc_common_cause.value = sd_uc_common_cause

        self.mean_uc0_individualism_random_variable = mean_uc0_individualism_random_variable
        self.sd_uc_individualism_random_variable = sd_uc_individualism_random_variable
        self.mean_uc0_coordination_random_variable = mean_uc0_coordination_random_variable
        self.sd_uc_coordination_random_variable = sd_uc_coordination_random_variable
        self.mean_uc0_common_cause_random_variable = mean_uc0_common_cause_random_variable
        self.sd_uc_common_cause_random_variable = sd_uc_common_cause_random_variable

        self.unbounded_individualism_observed_values = unbounded_individualism_observed_values
        self.unbounded_coordination_observed_values = unbounded_coordination_observed_values
        self.unbounded_common_cause_observed_values = unbounded_common_cause_observed_values

        self.initial_samples = initial_samples

    def draw_samples(
            self, seed: Optional[int], num_series: int
    ) -> SigmoidGaussianCoordinationSamples3D:
        """
        Draw coordination samples. A sample is a time series of coordination.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if either mean_uc0 or sd_uc is None.
        @return: coordination samples. One coordination series per row.
        """
        super().draw_samples(seed, num_series)

        if self.parameters.mean_uc0_individualism.value is None:
            raise ValueError(
                f"Value of {self.parameters.mean_uc0_individualism.uuid} is undefined.")

        if self.parameters.sd_uc_individualism.value is None:
            raise ValueError(f"Value of {self.parameters.sd_uc_individualism.uuid} is undefined.")

        if self.parameters.mean_uc0_coordination.value is None:
            raise ValueError(
                f"Value of {self.parameters.mean_uc0_coordination.uuid} is undefined.")

        if self.parameters.sd_uc_coordination.value is None:
            raise ValueError(f"Value of {self.parameters.sd_uc_coordination.uuid} is undefined.")

        if self.parameters.mean_uc0_common_cause.value is None:
            raise ValueError(
                f"Value of {self.parameters.mean_uc0_common_cause.uuid} is undefined.")

        if self.parameters.sd_uc_common_cause.value is None:
            raise ValueError(f"Value of {self.parameters.sd_uc_common_cause.uuid} is undefined.")

        mean_uc0 = np.array([self.parameters.mean_uc0_individualism.value,
                             self.parameters.mean_uc0_coordination.value,
                             self.parameters.mean_uc0_common_cause.value])[:, None]

        sd_uc = np.array([self.parameters.sd_uc_individualism.value,
                          self.parameters.sd_uc_coordination.value,
                          self.parameters.sd_uc_common_cause.value])[:, None]

        if self.initial_samples is not None:
            if self.initial_samples.shape[0] != num_series:
                raise ValueError(
                    f"The number of series {num_series} does not match the number of sampled "
                    f"series ({self.initial_samples.shape[0]}) in the provided unbounded "
                    f"coordination samples."
                )

            if self.initial_samples.shape[-1] > self.num_time_steps:
                raise ValueError(
                    f"The number of time steps ({self.initial_samples.shape[-1]}) in the provided "
                    f"unbounded coordination samples is larger than the requested number of time "
                    f"steps ({self.num_time_steps})."
                )

            dt = self.num_time_steps
            uc0 = self.initial_samples[..., -1]
        else:
            dt = self.num_time_steps
            uc0 = mean_uc0

        logging.info(
            f"Drawing {self.__class__.__name__} with {self.num_time_steps} time "
            f"steps."
        )

        if dt > 0:
            unbounded_coordination = (
                    norm(loc=0, scale=1).rvs(size=(num_series, 3, dt))
                    * sd_uc
            )
            unbounded_coordination[..., 0] += uc0
            unbounded_coordination = unbounded_coordination.cumsum(axis=-1)

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
        coordination = coordination / coordination.sum(axis=1, keepdims=True)

        return SigmoidGaussianCoordinationSamples3D(
            unbounded_coordination=unbounded_coordination, coordination=coordination
        )

    def create_random_variables(self):
        """
        Creates parameters and coordination variables in a PyMC model.
        """

        with self.pymc_model:
            if self.mean_uc0_individualism_random_variable is None:
                self.mean_uc0_individualism_random_variable = pm.Normal(
                    name=self.parameters.mean_uc0_individualism.uuid,
                    mu=self.parameters.mean_uc0_individualism.prior.mean,
                    sigma=self.parameters.mean_uc0_individualism.prior.sd,
                    size=1,
                    observed=adjust_dimensions(self.parameters.mean_uc0_individualism.value,
                                               num_rows=1),
                )

            if self.sd_uc_individualism_random_variable is None:
                self.sd_uc_individualism_random_variable = pm.HalfNormal(
                    name=self.parameters.sd_uc_individualism.uuid,
                    sigma=self.parameters.sd_uc_individualism.prior.sd,
                    size=1,
                    observed=adjust_dimensions(self.parameters.sd_uc_individualism.value,
                                               num_rows=1),
                )

            if self.mean_uc0_coordination_random_variable is None:
                self.mean_uc0_coordination_random_variable = pm.Normal(
                    name=self.parameters.mean_uc0_coordination.uuid,
                    mu=self.parameters.mean_uc0_coordination.prior.mean,
                    sigma=self.parameters.mean_uc0_coordination.prior.sd,
                    size=1,
                    observed=adjust_dimensions(self.parameters.mean_uc0_coordination.value,
                                               num_rows=1),
                )

            if self.sd_uc_coordination_random_variable is None:
                self.sd_uc_coordination_random_variable = pm.HalfNormal(
                    name=self.parameters.sd_uc_coordination.uuid,
                    sigma=self.parameters.sd_uc_coordination.prior.sd,
                    size=1,
                    observed=adjust_dimensions(self.parameters.sd_uc_coordination.value,
                                               num_rows=1),
                )

            if self.mean_uc0_common_cause_random_variable is None:
                self.mean_uc0_common_cause_random_variable = pm.Normal(
                    name=self.parameters.mean_uc0_common_cause.uuid,
                    mu=self.parameters.mean_uc0_common_cause.prior.mean,
                    sigma=self.parameters.mean_uc0_common_cause.prior.sd,
                    size=1,
                    observed=adjust_dimensions(self.parameters.mean_uc0_common_cause.value,
                                               num_rows=1),
                )

            if self.sd_uc_common_cause_random_variable is None:
                self.sd_uc_common_cause_random_variable = pm.HalfNormal(
                    name=self.parameters.sd_uc_common_cause.uuid,
                    sigma=self.parameters.sd_uc_common_cause.prior.sd,
                    size=1,
                    observed=adjust_dimensions(self.parameters.sd_uc_common_cause.value,
                                               num_rows=1),
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

                if self.facets_axis_name not in self.pymc_model.coords:
                    self.pymc_model.add_coord(
                        name=self.facets_axis_name,
                        values=["individualism", "coordination", "common_cause"]
                    )

                # Create variables
                prior_individualism = pm.Normal.dist(
                    mu=self.mean_uc0_individualism_random_variable,
                    sigma=self.sd_uc_individualism_random_variable,
                )
                unbounded_individualism_facet = pm.GaussianRandomWalk(
                    name="unbounded_individualism_facet",
                    init_dist=prior_individualism,
                    sigma=self.sd_uc_individualism_random_variable,
                    dims=[self.time_axis_name],
                    observed=self.unbounded_individualism_observed_values,
                )

                prior_coordination = pm.Normal.dist(
                    mu=self.mean_uc0_coordination_random_variable,
                    sigma=self.sd_uc_coordination_random_variable,
                )
                unbounded_coordination_facet = pm.GaussianRandomWalk(
                    name="unbounded_coordination_facet",
                    init_dist=prior_coordination,
                    sigma=self.sd_uc_coordination_random_variable,
                    dims=[self.time_axis_name],
                    observed=self.unbounded_coordination_observed_values,
                )

                prior_common_cause = pm.Normal.dist(
                    mu=self.mean_uc0_common_cause_random_variable,
                    sigma=self.sd_uc_common_cause_random_variable,
                )
                unbounded_common_cause_facet = pm.GaussianRandomWalk(
                    name="unbounded_common_cause_facet",
                    init_dist=prior_common_cause,
                    sigma=self.sd_uc_common_cause_random_variable,
                    dims=[self.time_axis_name],
                    observed=self.unbounded_common_cause_observed_values,
                )

                unbounded_coordination = pm.Deterministic(
                    name="unbounded_coordination",
                    var=ptt.stack([
                        unbounded_individualism_facet,
                        unbounded_coordination_facet,
                        unbounded_common_cause_facet
                    ], axis=0),
                    dims=[self.facets_axis_name, self.time_axis_name],
                )

                self.coordination_random_variable = pm.Deterministic(
                    name=self.uuid,
                    var=pm.math.sigmoid(unbounded_coordination) / pm.math.sigmoid(
                        unbounded_coordination).sum(axis=0)[None, :],
                    dims=[self.facets_axis_name, self.time_axis_name],
                )

    @property
    def facets_axis_name(self) -> str:
        return f"{self.uuid}_facets"


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SigmoidGaussianCoordinationParameters3D(ModuleParameters):
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
        self.mean_uc0_individualism = Parameter(
            uuid=f"{module_uuid}_mean_uc0_individualism",
            prior=NormalParameterPrior(mean=mean_mean_uc0, sd=sd_mean_uc0),
        )
        self.sd_uc_individualism = Parameter(
            uuid=f"{module_uuid}_sd_uc_individualism",
            prior=HalfNormalParameterPrior(sd_sd_uc),
        )
        self.mean_uc0_coordination = Parameter(
            uuid=f"{module_uuid}_mean_uc0_coordination",
            prior=NormalParameterPrior(mean=mean_mean_uc0, sd=sd_mean_uc0),
        )
        self.sd_uc_coordination = Parameter(
            uuid=f"{module_uuid}_sd_uc_coordination",
            prior=HalfNormalParameterPrior(sd_sd_uc),
        )
        self.mean_uc0_common_cause = Parameter(
            uuid=f"{module_uuid}_mean_uc0_common_cause",
            prior=NormalParameterPrior(mean=mean_mean_uc0, sd=sd_mean_uc0),
        )
        self.sd_uc_common_cause = Parameter(
            uuid=f"{module_uuid}_sd_uc_common_cause",
            prior=HalfNormalParameterPrior(sd_sd_uc),
        )


class SigmoidGaussianCoordinationSamples3D(ModuleSamples):
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
