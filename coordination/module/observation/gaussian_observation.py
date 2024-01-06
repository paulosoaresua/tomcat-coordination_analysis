from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.common.utils import adjust_dimensions
from coordination.module.latent_component.latent_component import \
    LatentComponentSamples
from coordination.module.module import ModuleParameters, ModuleSamples
from coordination.module.observation.observation import Observation
from coordination.module.parametrization2 import (HalfNormalParameterPrior,
                                                  Parameter)

NORMALIZATION_PER_FEATURE = "norm_per_feature"
NORMALIZATION_PER_SUBJECT_AND_FEATURE = "norm_per_subject_and_feature"


class GaussianObservation(Observation, ABC):
    """
    This class represents an observation (O) from a latent system component (A) sampled from
    a Gaussian distribution centered on some transformation, g(.), of the latent component, i.e.,
    O ~ N(g(A), var_o).
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        num_subjects: int,
        dimension_size: int,
        sd_sd_o: np.ndarray,
        share_sd_o_across_subjects: bool,
        share_sd_o_across_dimensions: bool,
        normalization: Optional[str] = None,
        dimension_names: Optional[List[str]] = None,
        latent_component_samples: Optional[LatentComponentSamples] = None,
        latent_component_random_variable: Optional[pm.Distribution] = None,
        observation_random_variable: Optional[pm.Distribution] = None,
        sd_o_random_variable: Optional[pm.Distribution] = None,
        observed_values: Optional[TensorTypes] = None,
        sd_o: Optional[Union[float, np.ndarray]] = None,
    ):
        """
        Creates a Gaussian observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param normalization: type of normalization to apply to observed values if desired. Valid
            normalization values are: norm_per_feature or norm_per_subject_and_feature.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        @param observation_random_variable: observation random variable to be used in a
            call to create_random_variables. If not set, it will be created in such a call.
        @param latent_component_samples: latent component samples to be used in a call to
            draw_samples. This variable must be set before such a call.
        @param latent_component_random_variable: latent component random variable to be used in a
            call to create_random_variables. This variable must be set before such a call.
        @param sd_o_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param observed_values: observations for the latent component random variable. If a value
            is set, the variable is not latent anymore.
        @param sd_o: standard deviation that represents the noise in the observations. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        """

        # No need to set coordination terms because a Gaussian observation only depends on the
        # latent component. It does not depend on coordination directly.
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=GaussianObservationParameters(module_uuid=uuid, sd_sd_o=sd_sd_o),
            num_subjects=num_subjects,
            dimension_size=dimension_size,
            dimension_names=dimension_names,
            observation_random_variable=observation_random_variable,
            latent_component_samples=latent_component_samples,
            latent_component_random_variable=latent_component_random_variable,
            observed_values=observed_values,
        )
        self.parameters.sd_o.value = sd_o

        self.normalization = normalization
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_dimensions = share_sd_o_across_dimensions
        self.sd_o_random_variable = sd_o_random_variable

    @abstractmethod
    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Draws latent component samples using ancestral sampling and some blending strategy with
        coordination and different subjects. This method must be implemented by concrete
        subclasses.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if latent_component_samples is None.
        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if self.latent_component_samples is None:
            raise ValueError(
                "No latent component samples. Please call  "
                "before invoking the draw_samples method."
            )

        if self.parameters.sd_o.value is None:
            raise ValueError(f"Value of {self.parameters.sd_o.uuid} is undefined.")

    @abstractmethod
    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.

        @raise ValueError: if latent_component_random_variable is None.
        """
        super().create_random_variables()

        with self.pymc_model:
            if self.sd_o_random_variable is None:
                dim_subjects = (
                    1 if self.share_sd_o_across_subjects else self.num_subjects
                )
                dim_dimensions = (
                    1 if self.share_sd_o_across_dimensions else self.dimension_size
                )
                self.sd_o_random_variable = pm.HalfNormal(
                    name=self.parameters.sd_o.uuid,
                    sigma=adjust_dimensions(
                        self.parameters.sd_o.prior.sd,
                        num_rows=dim_subjects,
                        num_cols=dim_dimensions,
                    ),
                    size=(dim_subjects, dim_dimensions),
                    observed=adjust_dimensions(
                        self.parameters.sd_o.value,
                        num_rows=dim_subjects,
                        num_cols=dim_dimensions,
                    ),
                )

        if self.observation_random_variable is None:
            if self.latent_component_random_variable is None:
                raise ValueError(
                    "Latent component variable is undefined. Please set "
                    "latent_component_random_variable before invoking the "
                    "create_random_variables method."
                )

    def _get_normalized_observation(self) -> np.ndarray:
        if self.normalization is None:
            print(f"Observations ({self.uuid}) will not be normalized.")
            return self.observed_values

        if self.normalization == NORMALIZATION_PER_FEATURE:
            print(f"Observations ({self.uuid}) will be normalized per feature.")
            return self._normalize_observation_per_feature()

        if self.normalization == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            print(
                f"Observations ({self.uuid}) will be normalized per subject and feature."
            )
            return self._normalize_observation_per_subject_and_feature()

        raise ValueError(f"Normalization ({self.normalization}) is invalid.")

    @abstractmethod
    def _normalize_observation_per_subject_and_feature(self) -> np.ndarray:
        """
        Normalize observed values to have mean 0 and standard deviation 1 across time. The
        normalization is done individually per subject and feature.

        @return: normalized observation.
        """

    @abstractmethod
    def _normalize_observation_per_feature(self) -> np.ndarray:
        """
        Normalize observed values to have mean 0 and standard deviation 1 across time and subject.
        The normalization is done individually per feature.

        @return: normalized observation.
        """


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
        self.sd_o = Parameter(
            uuid=f"{module_uuid}_sd_o", prior=HalfNormalParameterPrior(sd_sd_o)
        )
