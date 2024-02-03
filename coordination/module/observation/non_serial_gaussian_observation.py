from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.common.utils import adjust_dimensions
from coordination.module.constants import (DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_OBSERVATION_DIMENSION_SIZE,
                                           DEFAULT_OBSERVATION_SD_PARAM,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS)
from coordination.module.latent_component.serial_gaussian_latent_component import \
    SerialGaussianLatentComponentSamples
from coordination.module.observation.gaussian_observation import \
    GaussianObservation
from coordination.module.observation.observation import ObservationSamples


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
        latent_component_samples: Optional[SerialGaussianLatentComponentSamples] = None,
        latent_component_random_variable: Optional[pm.Distribution] = None,
        sd_o_random_variable: Optional[pm.Distribution] = None,
        time_steps_in_coordination_scale: Optional[np.array] = None,
        observed_values: Optional[TensorTypes] = None,
        sd_o: Optional[Union[float, np.ndarray]] = None,
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
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the observation scale.
        @param observed_values: observations for the latent component random variable. If a value
            is set, the variable is not latent anymore.
        @param sd_o: standard deviation that represents the noise in the observations. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
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
            sd_o=sd_o,
        )

        self.subject_names = subject_names
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

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

    def draw_samples(self, seed: Optional[int], num_series: int) -> ObservationSamples:
        """
        Draws observation samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: observation samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        dim_sd_o_subjects = 1 if self.share_sd_o_across_subjects else self.num_subjects
        dim_sd_o_dimensions = (
            1 if self.share_sd_o_across_dimensions else self.dimension_size
        )

        if (
            isinstance(self.parameters.sd_o.value, np.ndarray)
            and self.parameters.sd_o.value.ndim == 3
        ):
            # A different value per series. We expect it's already in the correct dimensions.
            sd_o = self.parameters.sd_o.value
            print("SHAPE")
            print(sd_o.shape)
        else:
            sd_o = adjust_dimensions(
                self.parameters.sd_o.value,
                num_rows=dim_sd_o_subjects,
                num_cols=dim_sd_o_dimensions,
            )
            if self.share_sd_o_across_subjects:
                sd_o = sd_o.repeat(self.num_subjects, axis=0)

            sd_o = sd_o[None, :].repeat(num_series, axis=0)

        # Broadcast across time
        sd_o = sd_o[:, None]
        print("SHAPE AFTER TIME")
        print(sd_o.shape)
        print("SHAPE LATENT")
        print(self.latent_component_samples.values)

        sampled_values = norm(loc=self.latent_component_samples.values, scale=sd_o).rvs(
            size=self.latent_component_samples.values.shape
        )

        return ObservationSamples(
            sampled_values,
            self.latent_component_samples.time_steps_in_coordination_scale,
        )

    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.
        """
        super().create_random_variables()

        if self.observation_random_variable is not None:
            return

        # subject x dimension x time (broadcast across time). If share_sd_o_across_subjects = True,
        # it is also broadcast across subjects (first axis has size 1)
        sd_o = self.sd_o_random_variable[..., None]

        logging.info(
            f"Fitting {self.__class__.__name__} with "
            f"{len(self.time_steps_in_coordination_scale)} time steps."
        )

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
