from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.parametrization2 import (Parameter,
                                                  HalfNormalParameterPrior)
from coordination.module.module import Module, ModuleParameters, ModuleSamples
from coordination.module.coordination2 import CoordinationSamples
from coordination.module.latent_component import LatentComponentSamples


class Observation(ABC, Module):
    """
    This class represents an observation (O) from a latent system component (A). Observations are
    evidence to the model. This implementation samples observation from a Gaussian distribution
     centered on some transformation, g(.),  of the latent components, i.e., O ~ N(g(A), var_o).
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 sd_sd_o: np.ndarray,
                 share_sd_o_across_subjects: bool,
                 share_sd_o_across_dimensions: bool,
                 dimension_names: Optional[List[str]] = None):
        """
        Creates an observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        """
        super().__init__()

        # If a parameter is shared across dimensions, we only have one parameter to infer.
        dim_sd_o_dimensions = 1 if share_sd_o_across_dimensions else dimension_size

        # Check if the values passed as hyperparameter are in agreement with the dimension of the
        # variables that we need to infer. Parameters usually have dimensions: num subjects x
        # dimension size, but that changes depending on the sharing options.
        if share_sd_o_across_subjects:
            assert (dim_sd_o_dimensions,) == sd_sd_o.shape
        else:
            assert (num_subjects, dim_sd_o_dimensions) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_dimensions = share_sd_o_across_dimensions
        self.dimension_names = dimension_names

        self.parameters = ObservationParameters(module_uuid=uuid,
                                                sd_sd_o=sd_sd_o)

    @abstractmethod
    def draw_samples(self,
                     seed: Optional[int],
                     coordination: CoordinationSamples = None,
                     latent_component: LatentComponentSamples = None,
                     **kwargs) -> ObservationSamples:
        """
        Draws observation samples using ancestral sampling.

        @param coordination: sampled coordination values.
        @param latent_component: sampled latent component values.
        @param seed: random seed for reproducibility.
        @raise ValueError: if either coordination or latent_component is None.
        @param kwargs: extra arguments to be defined by subclasses.

        @return: latent component samples for each coordination series.
        """
        if coordination is None:
            raise ValueError(f"No coordination samples.")

        if latent_component is None:
            raise ValueError(f"No latent component samples.")

        pass

    def _check_parameter_dimensionality_consistency(self):
        """
        Check if their dimensionality is consistent with the sharing options.
        """

        if self.parameters.mean_a0.value is None:
            raise ValueError("Initial mean parameter value is undefined.")

        if self.parameters.sd_aa.value is None:
            raise ValueError("Transition standard deviation parameter value is undefined.")

        dim_mean_a0_dimensions = 1 if self.share_mean_a0_across_dimensions else self.dimension_size
        if self.share_mean_a0_across_subjects:
            assert (dim_mean_a0_dimensions,) == self.parameters.mean_a0.value.shape
        else:
            assert (self.num_subjects,
                    dim_mean_a0_dimensions) == self.parameters.mean_a0.value.shape

        dim_sd_aa_dimensions = 1 if self.share_sd_aa_across_dimensions else self.dimension_size
        if self.share_sd_aa_across_subjects:
            assert (dim_sd_aa_dimensions,) == self.parameters.sd_aa.value.shape
        else:
            assert (self.num_subjects, dim_sd_aa_dimensions) == self.parameters.sd_aa.value.shape

    @abstractmethod
    def update_pymc_model(
            self,
            pymc_model: pm.Model,
            latent_component: pm.Distribution = None,
            observed_values: Optional[TensorTypes] = None,
            sd_o: Optional[pm.Distribution] = None,
            **kwargs) -> Tuple[
        Union[TensorTypes, pm.Distribution], ...]:
        """
        Creates parameters and observation variables in a PyMC model.

        @param pymc_model: model definition in pymc.
        @param latent_component: latent random variable representing a time series of latent
            component values.
        @param observed_values: latent component values if one wants to fix them. This will treat
            the latent component as known and constant. This is not the value of an observation
            component, but the latent component itself.
        @param sd_o: standard deviation of the latent component Gaussian transition distribution
            if previously defined outside of the component. This is useful if one wants to share this
            across different components.
        @param kwargs: extra parameters to be used by child classes.
        @return: random variables created in the PyMC model associated with the observation.
        """

        if latent_component is None:
            raise ValueError("Latent component variable is undefined.")

        pass

    def _create_emission_standard_deviation_variable(self) -> pm.Distribution:
        """
        Creates a latent variable for the standard deviation of the emission distribution. We
        assume independence between the individual parameters per subject and dimension and sample
        them from a multivariate Half-Gaussian.

        @return: a latent variable with a Half-Gaussian prior.
        """

        dim_sd_o_dimensions = 1 if self.share_sd_o_across_dimensions else self.dimension_size

        if self.share_sd_o_across_subjects:
            # When shared across subjects, only one parameter per dimension is needed.
            sd_o = pm.HalfNormal(name=self.parameters.sd_o.uuid,
                                  sigma=self.parameters.sd_o.prior.sd,
                                  size=dim_sd_o_dimensions,
                                  observed=self.parameters.sd_o.value)
        else:
            # Different parameters per subject and dimension.
            sd_o = pm.HalfNormal(name=self.parameters.sd_o.uuid,
                                  sigma=self.parameters.sd_o.prior.sd,
                                  size=(self.num_subjects, dim_sd_o_dimensions),
                                  observed=self.parameters.sd_o.value)

        return sd_o


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class ObservationParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of an observation module.
    """

    def __init__(self, module_uuid: str, sd_sd_o: np.ndarray):
        """
        Creates an object to store observation parameter info.

        @param module_uuid: unique ID of the observation module.
        @param sd_sd_o: standard deviation of the hyper-prior of the standard deviation used in
            the Gaussian emission distribution.
        """
        self.sd_o = Parameter(uuid=f"{module_uuid}_sd_o",
                              prior=HalfNormalParameterPrior(sd_sd_o))


class ObservationSamples(ModuleSamples):
    """
    This class stores samples generated by an observation module.
    """

    def __init__(self,
                 values: Union[List[np.ndarray], np.ndarray]):
        """
        Creates an object to store observation samples.

        @param values: sampled values of the observation. For serial observations, this will be
        a list of time series of values of different sizes. For non-serial observations, this will
        be a tensor as the number of observations in time do not change for different sampled time
        series.
        """
        super().__init__(values)
