from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.common.utils import set_random_seed
from coordination.module.latent_component import LatentComponentSamples
from coordination.module.observation2 import Observation, ObservationSamples, ObservationParameters


class GaussianObservation(ABC, Observation):
    """
    This class represents an observation (O) from a latent system component (A) sampled from
    a Gaussian distribution centered on some transformation, g(.), of the latent component, i.e.,
    O ~ N(g(A), var_o).
    """

    def __init__(self,
                 pymc_model: pm.Model,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 sd_sd_o: np.ndarray,
                 share_sd_o_across_subjects: bool,
                 share_sd_o_across_dimensions: bool,
                 dimension_names: Optional[List[str]] = None,
                 latent_component_samples: LatentComponentSamples = None,
                 latent_component_random_variable: pm.Distribution = None,
                 observation_random_variable: pm.Distribution = None,
                 sd_o_random_variable: pm.Distribution = None,
                 observed_values: TensorTypes = None):
        """
        Creates a Gaussian observation.

        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        @param observation_random_variable: observation random variable to be used in a
            call to update_pymc_model. If not set, it will be created in such a call.
        @param latent_component_samples: latent component samples to be used in a call to
            draw_samples. This variable must be set before such a call.
        @param latent_component_random_variable: latent component random variable to be used in a
            call to update_pymc_model. This variable must be set before such a call.
        @param sd_o_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param observed_values: observations for the latent component random variable. If a value
            is set, the variable is not latent anymore.
        """

        # No need to set coordination terms because a Gaussian observation only depends on the
        # latent component. It does not depend on coordination directly.
        super(Observation).__init__(pymc_model=pymc_model,
                                    uuid=uuid,
                                    num_subjects=num_subjects,
                                    dimension_size=dimension_size,
                                    dimension_names=dimension_names,
                                    coordination_samples=None,
                                    coordination_random_variable=None,
                                    observation_random_variable=observation_random_variable,
                                    observed_values=observed_values)

        # If a parameter is shared across dimensions, we only have one parameter to infer.
        dim_sd_o_dimensions = 1 if share_sd_o_across_dimensions else dimension_size

        # Check if the values passed as hyperparameter are in agreement with the dimension of the
        # variables that we need to infer. Parameters usually have dimensions: num subjects x
        # dimension size, but that changes depending on the sharing options.
        # TODO: replace asserts with ValueError
        if share_sd_o_across_subjects:
            assert (dim_sd_o_dimensions,) == sd_sd_o.shape
        else:
            assert (num_subjects, dim_sd_o_dimensions) == sd_sd_o.shape

        self.dimension_size = dimension_size
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_dimensions = share_sd_o_across_dimensions
        self.latent_component_samples = latent_component_samples
        self.latent_component_random_variable = latent_component_random_variable
        self.sd_o_random_variable = sd_o_random_variable

        self.parameters = GaussianObservationParameters(module_uuid=uuid,
                                                        sd_sd_o=sd_sd_o)

    @abstractmethod
    def draw_samples(self, seed: Optional[int]) -> ObservationSamples:
        """
        Draws latent component samples using ancestral sampling and some blending strategy with
        coordination and different subjects. This method must be implemented by concrete
        subclasses.

        @param seed: random seed for reproducibility.
        @raise ValueError: if coordination is None.
        @return: latent component samples for each coordination series.
        """
        super(Observation).draw_samples(seed)

        if self.latent_component_samples is None:
            raise ValueError("No latent component samples. Please call  "
                             "before invoking the draw_samples method.")

    def _check_parameter_dimensionality_consistency(self):
        """
        Check if their dimensionality is consistent with the sharing options.

        @raise ValueError: if sd_o has undefined values.
        """

        if self.parameters.sd_o.value is None:
            raise ValueError("Emission standard deviation parameter value is undefined.")

        dim_sd_o_dimensions = 1 if self.share_sd_o_across_dimensions else self.dimension_size
        if self.share_sd_o_across_subjects:
            assert (dim_sd_o_dimensions,) == self.parameters.sd_o.value.shape
        else:
            assert (self.num_subjects, dim_sd_o_dimensions) == self.parameters.sd_o.value.shape

    @abstractmethod
    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.

        @raise ValueError: if latent_component_random_variable is None.
        """
        super(Observation).update_pymc_model(pymc_model, observed_values)

        with self.pymc_model:
            if self.sd_o_random_variable is None:
                self.sd_o_random_variable = self._create_emission_standard_deviation_variable()

        if self.observation_random_variable is None:
            if self.latent_component_random_variable is None:
                raise ValueError("Latent component variable is undefined. Please set "
                                 "latent_component_random_variable before invoking the "
                                 "update_pymc_model method.")

    def _create_emission_standard_deviation_variable(self) -> pm.Distribution:
        """
        Creates a latent variable for the standard deviation of the emission distribution. We
        assume independence between the individual parameters per subject and dimension and sample
        them from a multivariate Half-Gaussian.

        @return: a latent variable with a Half-Gaussian prior.
        """

        dim_sd_o_dimensions = 1 if self.share_sd_o_across_dimensions else self.dimension_size

        with self.pymc_model:
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


class GaussianObservationParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of an observation module.
    """

    def __init__(self, module_uuid: str, sd_sd_o: np.ndarray):
        """
        Creates an object to store Gaussian observation parameter info.

        @param module_uuid: unique ID of the observation module.
        @param sd_sd_o: standard deviation of the hyper-prior of the standard deviation used in
            the Gaussian emission distribution.
        """
        self.sd_o = Parameter(uuid=f"{module_uuid}_sd_o",
                              prior=HalfNormalParameterPrior(sd_sd_o))
