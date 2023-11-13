from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.common.utils import set_random_seed
from coordination.module.serial_latent_component import SerialLatentComponentSamples
from coordination.module.gaussian_observation import GaussianObservation, \
    GaussianObservationParameters
from coordination.module.module import ModuleSamples
from coordination.module.constants import (DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_OBSERVATION_DIMENSION_SIZE,
                                           DEFAULT_SELF_DEPENDENCY,
                                           DEFAULT_OBSERVATION_SD_PARAM,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS)


class SerialGaussianObservation(GaussianObservation):
    """
    This class represents a Gaussian observation where there's only one observation per subject
    at a time in the module's scale, and latent components are influenced in a pair-wised manner.
    """

    def __init__(self,
                 uuid: str,
                 pymc_model: pm.Model,
                 num_subjects: int = DEFAULT_NUM_SUBJECTS,
                 dimension_size: int = DEFAULT_OBSERVATION_DIMENSION_SIZE,
                 sd_sd_o: np.ndarray = DEFAULT_OBSERVATION_SD_PARAM,
                 share_sd_o_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
                 share_sd_o_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
                 dimension_names: Optional[List[str]] = None,
                 observation_random_variable: Optional[pm.Distribution] = None,
                 latent_component_samples: Optional[SerialLatentComponentSamples] = None,
                 latent_component_random_variable: Optional[pm.Distribution] = None,
                 sd_o_random_variable: Optional[pm.Distribution] = None,
                 subject_indices: Optional[np.ndarray] = None,
                 observed_values: Optional[TensorTypes] = None):
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
        @param subject_indices: array of numbers indicating which subject is associated to the
            observation at every time step (e.g. the current speaker for a speech observation).
            In serial observations, only one subject's observation exists at a time. This
            array indicates which user that is. This array contains no gaps. The size of the array
            is the number of observed latent component in time, i.e., observation time
            indices with an associated subject.
        @param observed_values: observations for the latent component random variable. If a value
            is set, the variable is not latent anymore.
        """
        super().__init__(uuid=uuid,
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
                         observed_values=observed_values)

        self.subject_indices = subject_indices

    def draw_samples(self, seed: Optional[int],
                     num_series: int) -> SerialGaussianObservationSamples:
        """
        Draws observation samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: observation samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        self._check_parameter_dimensionality_consistency()

        observation_series = []
        for i in range(self.latent_component_samples.num_series):
            # Adjust dimensions according to parameter sharing specification
            if self.share_sd_o_across_subjects:
                # Broadcast across time
                sd = self.parameters.sd_o.value[:, None]
            else:
                sd = self.parameters.sd_o.value[subjects[i]].T

            samples = norm(loc=self.latent_component_samples.values[i], scale=sd).rvs(
                size=self.latent_component_samples.values[i].shape)
            observation_series.append(samples)

        return SerialGaussianObservationSamples(observation_series,
                                                self.latent_component_samples.subjects)

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
            sd_o = self.sd_o_random_variable[self.subject_indices].transpose()  # dimension x time

        if self.share_sd_o_across_dimensions:
            sd_o = sd_o.repeat(self.dimension_size, axis=0)

        # Add coordinates to the model
        if self.dimension_axis_name not in self.pymc_model.coords:
            self.pymc_model.add_coord(name=self.dimension_axis_name,
                                      values=self.dimension_names)

        if self.time_axis_name not in self.pymc_model.coords:
            self.pymc_model.add_coord(name=self.time_axis_name,
                                      values=np.arange(len(self.subject_indices)))

        with self.pymc_model:
            self.observation_random_variable = pm.Normal(
                name=self.uuid,
                mu=self.latent_component_random_variable,
                sigma=sd_o,
                dims=[self.dimension_axis_name, self.time_axis_name],
                observed=self.observed_values
            )


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SerialGaussianObservationSamples(ModuleSamples):

    def __init__(self,
                 values: List[np.ndarray],
                 subjects: List[np.ndarray]):
        """
        Creates an object to store samples and associated subjects in time.

        @param values: sampled observation values. This is a list of time series of values of
        different sizes because each sampled series may have a different sparsity level.
        @param subjects: number indicating which subject is associated to the observation at every
        time step (e.g. the current speaker for a speech component).
        """
        super().__init__(values)

        self.subjects = subjects
