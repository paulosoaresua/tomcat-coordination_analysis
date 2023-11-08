from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.parametrization2 import (Parameter,
                                                  HalfNormalParameterPrior,
                                                  NormalParameterPrior)
from coordination.module.coordination2 import CoordinationSamples
from coordination.module.module import Module, ModuleSamples, ModuleParameters


class LatentComponent(ABC, Module):
    """
    This class represents a latent system component. A latent system component is directly affected
    by coordination which controls to what extend one the latent component from one subject
    influences the same component in another subject in the future.

    We assume latent components evolve as a Gaussian random walk with mean defined by some blending
    strategy that takes coordination and past latent values from other subjects into consideration.

    A latent component can have a function f that transforms that mean in someway. For instance,
    if coordination manifests in an anti-symmetric way, f can be set to f(x) = -x.
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_a: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_dimensions: bool,
                 share_sd_a_across_subjects: bool,
                 share_sd_a_across_dimensions: bool,
                 dimension_names: Optional[List[str]] = None):
        """
        Creates a latent component module.

        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param self_dependent: whether the latent variables in the component are tied to the
            past values from the same subject. If False, coordination will blend the previous
            latent value of a different subject with the value of the component at time t = 0 for
            the current subject (the latent component's prior for that subject).
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
            latent component).
        @param sd_mean_a0: std of the hyper-prior of mu_a0.
        @param sd_sd_a: std of the hyper-prior of sigma_a (std of the Gaussian random walk of
            the latent component).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_a_across_subjects: whether to use the same sigma_a for all subjects.
        @param share_sd_a_across_dimensions: whether to use the same sigma_a for all dimensions.
        @param dimension_names: the names of each dimension of the latent component. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        """
        super().__init__()

        # If a parameter is shared across dimensions, we only have one parameter to infer.
        dim_mean_a0_dimensions = 1 if share_mean_a0_across_dimensions else dimension_size
        dim_sd_a_dimensions = 1 if share_sd_a_across_dimensions else dimension_size

        # Check if the values passed as hyperparameter are in agreement with the dimension of the
        # variables that we need to infer. Parameters usually have dimensions: num subjects x
        # dimension size, but that changes depending on the sharing options.
        if share_mean_a0_across_subjects:
            assert (dim_mean_a0_dimensions,) == mean_mean_a0.shape
            assert (dim_mean_a0_dimensions,) == sd_mean_a0.shape
        else:
            assert (num_subjects, dim_mean_a0_dimensions) == mean_mean_a0.shape
            assert (num_subjects, dim_mean_a0_dimensions) == sd_mean_a0.shape

        if share_sd_a_across_subjects:
            assert (dim_sd_a_dimensions,) == sd_sd_a.shape
        else:
            assert (num_subjects, dim_sd_a_dimensions) == sd_sd_a.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.self_dependent = self_dependent
        self.share_mean_a0_across_subjects = share_mean_a0_across_subjects
        self.share_mean_a0_across_dimensions = share_mean_a0_across_dimensions
        self.share_sd_a_across_subjects = share_sd_a_across_subjects
        self.share_sd_a_across_dimensions = share_sd_a_across_dimensions
        self.dimension_names = dimension_names

        self.parameters = LatentComponentParameters(module_uuid=uuid,
                                                    mean_mean_a0=mean_mean_a0,
                                                    sd_mean_a0=sd_mean_a0,
                                                    sd_sd_a=sd_sd_a)

        # Variables that must be set before a calling to draw_samples
        self.coordination_samples: CoordinationSamples = None

        # Variables that must be set before a calling to update_pymc_model
        self.coordination_random_variable: pm.Distribution = None

        # Variables that can be set before a calling to update_pymc_model
        self.mean_a0_random_variable: pm.Distribution = None
        self.sd_a_random_variable: pm.Distribution = None

    @abstractmethod
    def draw_samples(self, seed: Optional[int]) -> LatentComponentSamples:
        """
        Draws latent component samples using ancestral sampling and some blending strategy with
        coordination and different subjects. This method must be implemented by concrete
        subclasses.

        @param seed: random seed for reproducibility.
        @raise ValueError: if coordination is None.
        @return: latent component samples for each coordination series.
        """
        if self._coordination_samples is None:
            raise ValueError("No coordination samples. Please call set_coordination_samples "
                             "before invoking the draw_samples method.")

        pass

    def _check_parameter_dimensionality_consistency(self):
        """
        Check if their dimensionality is consistent with the sharing options.
        """

        if self.parameters.mean_a0.value is None:
            raise ValueError(f"Value of {self.parameters.mean_a0.uuid} is undefined.")

        if self.parameters.sd_a.value is None:
            raise ValueError(f"Value of {self.parameters.sd_a.uuid} is undefined.")

        dim_mean_a0_dimensions = 1 if self.share_mean_a0_across_dimensions else self.dimension_size
        if self.share_mean_a0_across_subjects:
            assert (dim_mean_a0_dimensions,) == self.parameters.mean_a0.value.shape
        else:
            assert (self.num_subjects,
                    dim_mean_a0_dimensions) == self.parameters.mean_a0.value.shape

        dim_sd_a_dimensions = 1 if self.share_sd_a_across_dimensions else self.dimension_size
        if self.share_sd_a_across_subjects:
            assert (dim_sd_a_dimensions,) == self.parameters.sd_a.value.shape
        else:
            assert (self.num_subjects, dim_sd_a_dimensions) == self.parameters.sd_a.value.shape

    @abstractmethod
    def update_pymc_model(
            self,
            pymc_model: pm.Model,
            observed_values: Optional[Dict[str, TensorTypes]] = None
    ) -> Dict[str, Union[TensorTypes, pm.Distribution], ...]:
        """
        Creates parameters and latent component variables in a PyMC model.

        @param pymc_model: model definition in pymc.
        @param observed_values: latent component values if one wants to fix them. This will treat
        the latent component as known and constant. This is not the value of an observation
        component, but the latent component itself.
        @raise ValueError: if coordination_random_variable is None.
        @return: random variables created in the PyMC model associated with the latent component.
        """

        if self.coordination_random_variable is None:
            raise ValueError("Coordination variable is undefined. Please set "
                             "coordination_random_variable before invoking the update_pymc_model "
                             "method.")

        if self.mean_a0_random_variable is None:
            self.mean_a0_random_variable = self._create_initial_mean_variable()

        if self.sd_a_random_variable is None:
            self.sd_a_random_variable = self._create_transition_standard_deviation_variable()

        pass

    def _create_initial_mean_variable(self) -> pm.Distribution:
        """
        Creates a latent variable for the mean of the initial state. We assume independence between
        the individual parameters per subject and dimension and sample them from a multivariate
        Gaussian.

        @return: a latent variable with a Gaussian prior.
        """

        dim_mean_a0_dimensions = 1 if self.share_mean_a0_across_dimensions else self.dimension_size

        if self.share_mean_a0_across_subjects:
            # When shared across subjects, only one parameter per dimension is needed.
            mean_a0 = pm.Normal(name=self.parameters.mean_a0.uuid,
                                mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd,
                                size=dim_mean_a0_dimensions,
                                observed=self.parameters.mean_a0.value)
        else:
            # Different parameters per subject and dimension.
            mean_a0 = pm.Normal(name=self.parameters.mean_a0.uuid,
                                mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd,
                                size=(self.num_subjects, dim_mean_a0_dimensions),
                                observed=self.parameters.mean_a0.value)

        return mean_a0

    def _create_transition_standard_deviation_variable(self) -> pm.Distribution:
        """
        Creates a latent variable for the standard deviation of the state transition (Gaussian
        random walk). We assume independence between the individual parameters per subject and
        dimension and sample them from a multivariate Half-Gaussian.

        @return: a latent variable with a Half-Gaussian prior.
        """

        dim_sd_a_dimensions = 1 if self.share_sd_a_across_dimensions else self.dimension_size

        if self.share_sd_a_across_subjects:
            # When shared across subjects, only one parameter per dimension is needed.
            sd_a = pm.HalfNormal(name=self.parameters.sd_a.uuid,
                                 sigma=self.parameters.sd_a.prior.sd,
                                 size=dim_sd_a_dimensions,
                                 observed=self.parameters.sd_a.value)
        else:
            # Different parameters per subject and dimension.
            sd_a = pm.HalfNormal(name=self.parameters.sd_a.uuid,
                                 sigma=self.parameters.sd_a.prior.sd,
                                 size=(self.num_subjects, dim_sd_a_dimensions),
                                 observed=self.parameters.sd_a.value)

        return sd_a


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class LatentComponentParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of a latent component.
    """

    def __init__(self,
                 module_uuid: str,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_a: np.ndarray):
        """
        Creates an object to store latent component parameter info.

        @param module_uuid: unique ID of the latent component module.
        @param mean_mean_a0: mean of the hyper-prior of the mean at time t = 0.
        @param sd_mean_a0: standard deviation of the hyper-prior of the mean at time t = 0.
        @param sd_sd_a: standard deviation of the hyper-prior of the standard deviation used in
            the Gaussian random walk when transitioning from one time to the next.
        """
        self.mean_a0 = Parameter(uuid=f"{module_uuid}_mean_a0",
                                 prior=NormalParameterPrior(mean_mean_a0, sd_mean_a0))
        self.sd_a = Parameter(uuid=f"{module_uuid}_sd_a",
                              prior=HalfNormalParameterPrior(sd_sd_a))


class LatentComponentSamples(ModuleSamples):
    """
    This class stores samples generated by a latent component.
    """

    def __init__(self,
                 values: Union[List[np.ndarray], np.ndarray],
                 time_steps_in_coordination_scale: Union[List[np.ndarray], np.ndarray]):
        """
        Creates an object to store samples.

        @param values: sampled values of the latent component. For serial components, this will be
        a list of time series of values of different sizes. For non-serial components, this will be
        a tensor as the number of observations in time do not change for different sampled time
        series.
        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different time scale from the time scale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the latent component. For serial components, this will be a list of time series of
        indices of different sizes. For non-serial components, this will be a tensor as the number
        of observations in time do not change for different sampled time series.
        """
        super().__init__(values=values)

        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
