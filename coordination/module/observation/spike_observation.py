from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.common.utils import set_random_seed
from coordination.module.observation.observation import Observation
from coordination.module.module import ModuleSamples, ModuleParameters
from coordination.module.parametrization2 import Parameter, BetaParameterPrior
from coordination.module.constants import DEFAULT_SAMPLING_TIME_SCALE_DENSITY


class SpikeObservation(Observation):
    """
    This class represents a binary series of observations (O) sampled from a Bernoulli distribution
    with parameter (p) that depends on the value of coordination. The intuition is that high
    coordination is more likely to generate spikes in observations of this kind. This, this module
    is can be used to model binary data that is 1 whenever it is expected that coordination is
    high.
    """

    def __init__(self,
                 uuid: str,
                 pymc_model: pm.Model,
                 num_subjects: int,
                 a_p: float,
                 b_p: float,
                 dimension_name: str = None,
                 coordination_samples: Optional[ModuleSamples] = None,
                 coordination_random_variable: Optional[pm.Distribution] = None,
                 p_random_variable: Optional[pm.Distribution] = None,
                 observation_random_variable: Optional[pm.Distribution] = None,
                 sampling_time_scale_density: float = DEFAULT_SAMPLING_TIME_SCALE_DENSITY,
                 time_steps_in_coordination_scale: Optional[np.array] = None,
                 observed_values: Optional[TensorTypes] = None):
        """
        Creates a Gaussian observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param a_p: parameter a of the hyper-prior (Beta) of the parameter p (Bernoulli).
        @param b_p: parameter b of the hyper-prior (Beta) of the parameter p (Bernoulli).
        @param dimension_name: name of the single dimension of the observation module. If not
            informed, this will be 0.
        @param coordination_samples: coordination samples to be used in a call to draw_samples.
            This variable must be set before such a call.
        @param coordination_random_variable: coordination random variable to be used in a call to
            create_random_variables. This variable must be set before such a call.
        @param p_random_variable: random variable to be used in a call to
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
        """

        # No need to set latent component terms because a spike observation only depends on the
        # coordination. It does not depend on the latent component.
        super().__init__(uuid=uuid,
                         pymc_model=pymc_model,
                         parameters=SpikeObservationParameters(module_uuid=uuid,
                                                               a_p=a_p,
                                                               b_p=b_p),
                         num_subjects=num_subjects,
                         dimension_size=1,
                         dimension_names=[dimension_name],
                         coordination_samples=coordination_samples,
                         coordination_random_variable=coordination_random_variable,
                         observation_random_variable=observation_random_variable,
                         observed_values=observed_values)

        self.p_random_variable = p_random_variable
        self.sampling_time_scale_density = sampling_time_scale_density
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

    def draw_samples(self, seed: Optional[int], num_series: int) -> SpikeObservationSamples:
        """
        Draws spike observation samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if coordination is None.
        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if self.coordination_samples is None:
            raise ValueError("No coordination samples. Please set coordination_samples "
                             "before invoking the draw_samples method.")

        density_mask = bernoulli(p=self.sampling_time_scale_density).rvs(
            self.coordination_samples.values.shape)

        # Effectively observe links according to the values of coordination
        links = bernoulli(p=coordination * self.parameters.p.value).rvs(
            self.coordination_samples.values.shape)

        # Mask out spikes according to the required density.
        links *= density_mask

        time_steps = []
        for s in range(num_series):
            # A spike has a constant value of 1.
            # We store the time steps when that a spike was observed
            time_steps.append(np.array([t for t, l in enumerate(links[s]) if l == 1]))

        return SpikeObservationSamples(time_steps)

    def create_random_variables(self):
        """
        Creates parameters and observation variables in a PyMC model.

        @raise ValueError: if coordination_random_variable is None.
        """
        super().create_random_variables()

        if self.coordination_random_variable is None:
            raise ValueError("Coordination variable is undefined. Please set "
                             "coordination_random_variable before invoking the "
                             "create_random_variables method.")

        if self.p_random_variable is not None:
            return

        if self.time_steps_in_coordination_scale is None:
            raise ValueError("time_steps_in_coordination_scale is undefined.")

        with self.pymc_model:
            self.p_random_variable = pm.Beta(name=self.parameters.p.uuid,
                                             alpha=self.parameters.p.prior.a,
                                             beta=self.parameters.p.prior.b,
                                             size=1,
                                             observed=self.parameters.p.value)

            adjusted_prob = pm.Deterministic(
                f"{self.uuid}_adjusted_p",
                self.p_random_variable * self.coordination_random_variable[
                    self.time_steps_in_coordination_scale])

            self.observation_random_variable = pm.Bernoulli(self.uuid,
                                                            adjusted_prob,
                                                            dims=self.time_axis_name,
                                                            observed=self.observed_values)

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model. Overrides superclass.
        """
        super()._add_coordinates()

        self.pymc_model.add_coord(name=self.time_axis_name,
                                  values=self.time_steps_in_coordination_scale)


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SpikeObservationParameters(ModuleParameters):
    """
    This class stores values and hyper-priors of the parameters of the spike observation module.
    """

    def __init__(self, module_uuid: str, a_p: float, b_p: float):
        """
        Creates an object to store spike observation parameter info.

        @param module_uuid: unique ID of the observation module.
        @param a_p: parameter a of the hyper-prior (Beta) of the parameter p (Bernoulli).
        @param b_p: parameter b of the hyper-prior (Beta) of the parameter p (Bernoulli).
        """
        self.p = Parameter(uuid=f"{module_uuid}_p",
                           prior=BetaParameterPrior(a_p, b_p))


class SpikeObservationSamples(ModuleSamples):
    """
    This class stores samples generated by an spike observation.
    """

    def __init__(self,
                 time_steps_in_coordination_scale: List[np.ndarray]):
        """
        Creates an object to store samples.

        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different timescale from the timescale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the spike observation. This is a list of time series of values of different sizes
        because each sampled series may have a different sparsity level.
        """
        super().__init__(
            values=[np.ones(series.shape[0]) for series in time_steps_in_coordination_scale])

        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
