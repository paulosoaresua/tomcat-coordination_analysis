from __future__ import annotations

from typing import List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_OBSERVATION_DIMENSION_SIZE,
                                           DEFAULT_OBSERVATION_SD_PARAM,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS,
                                           DEFAULT_OBSERVATION_NORMALIZATION)
from coordination.module.latent_component.serial_gaussian_latent_component import \
    SerialGaussianLatentComponentSamples
from coordination.module.module import ModuleSamples
from coordination.module.observation.gaussian_observation import \
    GaussianObservation


class SerialGaussianObservation(GaussianObservation):
    """
    This class represents a Gaussian observation where there's only one subject per observation
    at a time in the module's scale.
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
            normalize_observed_values: bool = DEFAULT_OBSERVATION_NORMALIZATION,
            dimension_names: Optional[List[str]] = None,
            observation_random_variable: Optional[pm.Distribution] = None,
            latent_component_samples: Optional[SerialGaussianLatentComponentSamples] = None,
            latent_component_random_variable: Optional[pm.Distribution] = None,
            sd_o_random_variable: Optional[pm.Distribution] = None,
            time_steps_in_coordination_scale: Optional[np.array] = None,
            subject_indices: Optional[np.ndarray] = None,
            observed_values: Optional[TensorTypes] = None
    ):
        """
        Creates a serial Gaussian observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param normalize_observed_values: whether to normalize observed_values before inference to
            have mean 0 and standard deviation 1 across time per subject and variable dimension.
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
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the observation scale.
        @param subject_indices: array of numbers indicating which subject is associated to the
            observation at every time step (e.g. the current speaker for a speech observation).
            In serial observations, only one subject's observation exists at a time. This
            array indicates which user that is. This array contains no gaps. The size of the array
            is the number of observed latent component in time, i.e., observation time
            indices with an associated subject.
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
            normalize_observed_values=normalize_observed_values,
            dimension_names=dimension_names,
            observation_random_variable=observation_random_variable,
            latent_component_samples=latent_component_samples,
            latent_component_random_variable=latent_component_random_variable,
            sd_o_random_variable=sd_o_random_variable,
            observed_values=observed_values
        )

        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.subject_indices = subject_indices

    def draw_samples(
            self, seed: Optional[int], num_series: int
    ) -> SerialGaussianObservationSamples:
        """
        Draws observation samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: observation samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        observation_series = []
        for i in range(self.latent_component_samples.num_series):
            # Adjust dimensions according to parameter sharing specification
            if self.share_sd_o_across_subjects:
                # Broadcast across time
                sd = self.parameters.sd_o.value[:, None]
            else:
                sd = self.parameters.sd_o.value[self.subject_indices[i]].T

            samples = norm(loc=self.latent_component_samples.values[i], scale=sd).rvs(
                size=self.latent_component_samples.values[i].shape
            )
            observation_series.append(samples)

        time_steps = self.latent_component_samples.time_steps_in_coordination_scale
        return SerialGaussianObservationSamples(
            values=observation_series,
            time_steps_in_coordination_scale=time_steps,
            subject_indices=self.latent_component_samples.subject_indices,
        )

    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.
        """
        super().create_random_variables()

        if self.observation_random_variable is not None:
            return

        if self.share_sd_o_across_subjects:
            # dimension x time = 1 (broadcast across time)
            sd_o = self.sd_o_random_variable[:, None]
        else:
            sd_o = self.sd_o_random_variable[
                self.subject_indices
            ].transpose()  # dimension x time

        if self.share_sd_o_across_dimensions:
            sd_o = sd_o.repeat(self.dimension_size, axis=0)

        with self.pymc_model:
            observation = self._normalize_observation() if self.normalize_observed_values \
                else self.observed_values
            self.observation_random_variable = pm.Normal(
                name=self.uuid,
                mu=self.latent_component_random_variable,
                sigma=sd_o,
                dims=[self.dimension_axis_name, self.time_axis_name],
                observed=observation,
            )

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model. Overrides superclass.

        @raise ValueError: if either subject_indices or time_steps_in_coordination_scale are
            undefined.
        """
        super()._add_coordinates()

        if self.subject_indices is None:
            raise ValueError("subject_indices is undefined.")

        if self.time_steps_in_coordination_scale is None:
            raise ValueError("time_steps_in_coordination_scale is undefined.")

        # Add information about which subject is associated with each timestep in the time
        # coordinate.
        self.pymc_model.add_coord(
            name=self.time_axis_name,
            values=[
                f"{sub}#{time}"
                for sub, time in zip(
                    self.subject_indices, self.time_steps_in_coordination_scale
                )
            ],
        )

    def _normalize_observation(self) -> np.ndarray:
        """
        Normalize observed values to have mean 0 and standard deviation 1 across time. The
        normalization is done individually per subject and variable dimension.

        @return: normalized observation.
        """

        normalized_values = np.zeros_like(self.observed_values)
        for subject in range(self.num_subjects):
            # Get observed values for a specific subject across time.
            idx = (self.subject_indices == subject)
            obs_per_subject = self.observed_values[:, idx]
            mean = np.mean(obs_per_subject, axis=-1, keepdims=True)  # mean across time
            std = np.std(obs_per_subject, axis=-1, keepdims=True)
            normalized_values[:, idx] = (obs_per_subject - mean) / std

        return normalized_values


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SerialGaussianObservationSamples(ModuleSamples):
    def __init__(
            self,
            values: List[np.ndarray],
            time_steps_in_coordination_scale: List[np.ndarray],
            subject_indices: List[np.ndarray],
    ):
        """
        Creates an object to store samples and associated subjects in time.

        @param values: sampled observation values. This is a list of time series of values of
            different sizes because each sampled series may have a different sparsity level.
        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
            sample. If the component is in a different timescale from the timescale used to compute
            coordination, this mapping will tell which value of coordination to map to each sampled
            value of the latent component.
        @param subject_indices: series indicating which subject is associated to the observation
            at every time step (e.g. the current speaker for a speech component).
        """
        super().__init__(values)

        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.subject_indices = subject_indices
