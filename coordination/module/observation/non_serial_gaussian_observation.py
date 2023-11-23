from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_OBSERVATION_DIMENSION_SIZE,
                                           DEFAULT_OBSERVATION_SD_PARAM,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS)
from coordination.module.latent_component.serial_latent_component import \
    SerialLatentComponentSamples
from coordination.module.module import ModuleSamples
from coordination.module.observation.gaussian_observation import \
    GaussianObservation


class NonSerialGaussianObservation(GaussianObservation):
    """
    This class represents a Gaussian observation where there are there are observations for all
    the subjects at each time in the module's scale.
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        num_subjects: int = DEFAULT_NUM_SUBJECTS,
        dimension_size: int = DEFAULT_OBSERVATION_DIMENSION_SIZE,
        sd_sd_o: np.ndarray = DEFAULT_OBSERVATION_SD_PARAM,
        share_sd_o_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
        share_sd_o_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
        dimension_names: Optional[List[str]] = None,
        subject_names: Optional[List[str]] = None,
        observation_random_variable: Optional[pm.Distribution] = None,
        latent_component_samples: Optional[SerialLatentComponentSamples] = None,
        latent_component_random_variable: Optional[pm.Distribution] = None,
        sd_o_random_variable: Optional[pm.Distribution] = None,
        observed_values: Optional[TensorTypes] = None,
    ):
        """
        Creates a non-serial Gaussian observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        @param subject_names: the names of each subject of the latent component. If not
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
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=dimension_size,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_subjects,
            share_sd_o_across_dimensions=share_sd_o_across_dimensions,
            dimension_names=dimension_names,
            observation_random_variable=observation_random_variable,
            latent_component_samples=latent_component_samples,
            latent_component_random_variable=latent_component_random_variable,
            sd_o_random_variable=sd_o_random_variable,
            observed_values=observed_values,
        )

        self.subject_names = subject_names

    @property
    def subject_coordinates(self) -> Union[List[str], np.ndarray]:
        """
        Gets a list of values representing the names of each subject.

        @return: a list of dimension names.
        """
        return (
            np.arange(self.num_subjects)
            if self.subject_names is None
            else self.subject_names
        )

    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Draws observation samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: observation samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        # Adjust dimensions according to parameter sharing specification
        if self.share_sd_o_across_subjects:
            # Broadcast across series, subjects and time
            sd = self.parameters.sd_o.value[None, None, :, None]
        else:
            # Broadcast across series and time
            sd = self.parameters.sd_o.value[None, :, :, None]

        sampled_values = norm(loc=self.latent_component_samples.values, scale=sd).rvs(
            size=self.latent_component_samples.values.shape
        )

        return ModuleSamples(sampled_values)

    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.
        """
        super().create_random_variables()

        if self.observation_random_variable is not None:
            return

        if self.share_sd_o_across_subjects:
            # subject x feature x time (broadcast across subject and time)
            sd_o = self.sd_o_random_variable[None, :, None]
        else:
            # subject x feature x time (broadcast across time)
            sd_o = self.sd_o_random_variable[:, :, None]

        with self.pymc_model:
            self.observation_random_variable = pm.Normal(
                name=self.uuid,
                mu=self.latent_component_random_variable,
                sigma=sd_o,
                dims=[
                    self.subject_axis_name,
                    self.dimension_axis_name,
                    self.time_axis_name,
                ],
                observed=self.observed_values,
            )

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model. Overrides superclass.
        """
        super()._add_coordinates()

        self.pymc_model.add_coord(
            name=self.subject_axis_name, values=self.subject_coordinates
        )
        self.pymc_model.add_coord(
            name=self.time_axis_name, values=self.time_steps_in_coordination_scale
        )
