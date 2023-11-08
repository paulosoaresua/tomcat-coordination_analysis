from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.common.utils import set_random_seed
from coordination.module.serial_latent_component import SerialLatentComponentSamples
from coordination.module.observation2 import Observation, ObservationSamples, ObservationParameters


class SerialObservation(Observation):
    """
    This class represents am observation where there's only one observation per subject
    at a time in the module's scale, and latent components are influenced in a pair-wised manner.
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
        Creates a serial observation.

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
        super().__init__(uuid=uuid,
                         num_subjects=num_subjects,
                         dimension_size=dimension_size,
                         sd_sd_o=sd_sd_o,
                         share_sd_o_across_subjects=share_sd_o_across_subjects,
                         share_sd_o_across_dimensions=share_sd_o_across_dimensions,
                         dimension_names=dimension_names)

    def draw_samples(self,
                     seed: Optional[int],
                     coordination: CoordinationSamples = None,
                     latent_component: SerialLatentComponentSamples = None,
                     **kwargs) -> LatentComponentSamples:
        """
        Draws observation samples using ancestral sampling.

        @param coordination: sampled coordination values.
        @param latent_component: sampled latent component values.
        @param seed: random seed for reproducibility.
        @param kwargs: extra arguments to be defined by subclasses.

        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, coordination, latent_component)

        self._check_parameter_dimensionality_consistency()

        # Generate samples
        set_random_seed(seed)

        observation_series = []
        for i in range(len(latent_component)):
            # Adjust dimensions according to parameter sharing specification
            if self.share_sd_o_across_subjects:
                # Broadcast across time
                sd = self.parameters.sd_o.value[:, None]
            else:
                sd = self.parameters.sd_o.value[subjects[i]].T

            samples = norm(loc=latent_component[i], scale=sd).rvs(size=latent_component[i].shape)
            observation_series.append(samples)

        return SerialObservationSamples(observation_series, latent_component.subjects)

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
            Precisely, observation and sd_o
        """

        with pymc_model:
            sd_o = self._create_emission_standard_deviation_variable() if sd_o is None else sd_o

        if self.share_sd_o_across_subjects:
            sd_o = sd_o[:, None]  # dimension x time = 1 (broadcast across time)
        else:
            sd_o = sd_o[subjects].transpose()  # dimension x time

        if self.share_sd_o_across_features:
            sd_o = sd_o.repeat(self.dimension_size, axis=0)

        dimension_axis_name = f"{self.uuid}_dimension"
        time_axis_name = f"{self.uuid}_time"

        with pymc_model:
            observation = pm.Normal(name=self.uuid,
                                    mu=latent_component,
                                    sigma=sd_o,
                                    dims=[dimension_axis_name, time_axis_name],
                                    observed=observed_values)

        return observation, sd_o


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SerialObservationSamples(ObservationSamples):

    def __init__(self,
                 values: List[np.ndarray],
                 subjects: List[np.ndarray]):
        """
        Creates an object to store samples.

        @param values: sampled observation values. This is a list of time series of values of
        different sizes because each sampled series may have a different sparsity level.
        @param subjects: number indicating which subject is associated to the observation at every
        time step (e.g. the current speaker for a speech component).
        """
        super().__init__(values)

        self.subjects = subjects
