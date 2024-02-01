from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pymc as pm
from scipy.stats import bernoulli, norm

from coordination.common.types import TensorTypes
from coordination.common.utils import adjust_dimensions
from coordination.module.constants import DEFAULT_SAMPLING_TIME_SCALE_DENSITY
from coordination.module.module import ModuleParameters, ModuleSamples
from coordination.module.observation.observation import Observation
from coordination.module.parametrization import HalfNormalParameterPrior, Parameter


class SpikeObservation(Observation):
    """
    This class represents a binary series of observations (O) sampled from a normal distribution
    centered in the coordination values with parameter (sd_s). The intuition is that high
    coordination is more likely to generate spikes in observations of this kind. The strength of
    the spike is controlled by sd_s. The larger, the smaller the influence. This, this module
    is can be used to model binary data that is 1 whenever it is expected that coordination is
    high.
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        num_subjects: int,
        sd_sd_s: float,
        dimension_name: str = None,
        coordination_samples: Optional[ModuleSamples] = None,
        coordination_random_variable: Optional[pm.Distribution] = None,
        sd_s_random_variable: Optional[pm.Distribution] = None,
        observation_random_variable: Optional[pm.Distribution] = None,
        sampling_time_scale_density: float = DEFAULT_SAMPLING_TIME_SCALE_DENSITY,
        time_steps_in_coordination_scale: Optional[np.array] = None,
        observed_values: Optional[TensorTypes] = None,
        sd_s: Optional[float] = None,
    ):
        """
        Creates a Gaussian observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param sd_sd_s: parameter a of the hyper-prior (HalfNormal) of the parameter sd_s.
        @param dimension_name: name of the single dimension of the observation module. If not
            informed, this will be 0.
        @param coordination_samples: coordination samples to be used in a call to draw_samples.
            This variable must be set before such a call.
        @param coordination_random_variable: coordination random variable to be used in a call to
            create_random_variables. This variable must be set before such a call.
        @param sd_s_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param observation_random_variable: observation random variable to be used in a
            call to create_random_variables. If not set, it will be created in such a call.
        @param sampling_time_scale_density: a number between 0 and 1 indicating the frequency in
            which we have observations. If 1, we have an observation is at every time in the
            coordination timescale. If 0.5, in average, only half of the time. The final number of
            observations is not a deterministic function of the time density, as the density is
            used as parameter os a Bernoulli distribution that determines whether we have an
            observation at a given time step or not.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the observation scale.
        @param observed_values: observations for the latent component random variable. If a value
            is set, the variable is not latent anymore.
        @param sd_s: parameter sd_s of the Normal distribution of the module. It needs to be given
            for sampling but not for inference if it needs to be inferred. If not provided now,
            it can be set later via the module parameters variable.
        """

        # No need to set latent component terms because a spike observation only depends on the
        # coordination. It does not depend on the latent component.
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=SpikeObservationParameters(module_uuid=uuid, sd_sd_s=sd_sd_s),
            num_subjects=num_subjects,
            dimension_size=1,
            dimension_names=[dimension_name],
            coordination_samples=coordination_samples,
            coordination_random_variable=coordination_random_variable,
            observation_random_variable=observation_random_variable,
            observed_values=observed_values,
        )
        self.parameters.sd_s.value = sd_s

        self.sd_s_random_variable = sd_s_random_variable
        self.sampling_time_scale_density = sampling_time_scale_density
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

    def draw_samples(
        self, seed: Optional[int], num_series: int
    ) -> SpikeObservationSamples:
        """
        Draws spike observation samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if coordination is None.
        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if self.coordination_samples is None:
            raise ValueError(
                "No coordination samples. Please set coordination_samples "
                "before invoking the draw_samples method."
            )

        time_steps = []
        for s in range(num_series):
            sd_s = self.parameters.sd_s.value
            if sd_s.ndim > 1:
                sd_s = sd_s[s]

            density_mask = bernoulli(p=self.sampling_time_scale_density).rvs(
                len(self.coordination_samples.values[s])
            )

            # Effectively observe links according to the values of coordination
            # links = bernoulli(
            #     p=self.coordination_samples.values[s] * p
            # ).rvs()
            links = norm(loc=self.coordination_samples.values[s], scale=sd_s).rvs()

            # Mask out spikes according to the required density.
            links *= density_mask
            # A spike has a constant value of 1.
            # We store the time steps when that a spike was observed
            # Simple rule to determine a spike here if we want to generate synthetic data.
            time_steps.append(np.array([t for t, l in enumerate(links) if l > 0.8]))

        return SpikeObservationSamples(time_steps)

    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.

        @raise ValueError: if coordination_random_variable is None.
        """
        super().create_random_variables()

        if self.coordination_random_variable is None:
            raise ValueError(
                "Coordination variable is undefined. Please set "
                "coordination_random_variable before invoking the "
                "create_random_variables method."
            )

        if self.p_random_variable is not None:
            return

        if self.time_steps_in_coordination_scale is None:
            raise ValueError("time_steps_in_coordination_scale is undefined.")

        logging.info(
            f"Fitting {self.__class__.__name__} with "
            f"{len(self.time_steps_in_coordination_scale)} time steps."
        )

        with self.pymc_model:
            self.sd_s_random_variable = pm.HalfNormal(
                name=self.parameters.sd_s.uuid,
                sigma=adjust_dimensions(self.parameters.sd_s.prior.sd, 1),
                size=1,
                observed=adjust_dimensions(self.parameters.sd_s.value, num_rows=1),
            )

            self.observation_random_variable = pm.Normal(
                self.uuid,
                mu=self.coordination_random_variable[
                    self.time_steps_in_coordination_scale
                ],
                sigma=self.sd_s_random_variable,
                dims=self.time_axis_name,
                observed=self.observed_values,
            )

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model. Overrides superclass.
        """
        super()._add_coordinates()

        self.pymc_model.add_coord(
            name=self.time_axis_name, values=self.time_steps_in_coordination_scale
        )


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SpikeObservationParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of the spike observation module.
    """

    def __init__(self, module_uuid: str, sd_sd_s: float):
        """
        Creates an object to store spike observation parameter info.

        @param module_uuid: unique ID of the observation module.
        @param sd_sd_s: parameter a of the hyper-prior (HalfNormal) of the parameter sd_s.
        """
        self.sd_s = Parameter(uuid=f"{module_uuid}_sd_s", prior=HalfNormalParameterPrior(sd_sd_s))


class SpikeObservationSamples(ModuleSamples):
    """
    This class stores samples generated by an spike observation.
    """

    def __init__(self, time_steps_in_coordination_scale: List[np.ndarray]):
        """
        Creates an object to store samples.

        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different timescale from the timescale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the spike observation. This is a list of time series of values of different sizes
        because each sampled series may have a different sparsity level.
        """
        super().__init__(
            values=[
                np.ones(series.shape[0]) for series in time_steps_in_coordination_scale
            ]
        )

        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
