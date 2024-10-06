from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.module import Module, ModuleParameters, ModuleSamples


class LatentComponent(ABC, Module):
    """
    This class represents an abstract latent system component. A latent system component is
    directly affected by coordination which controls to what extend one the latent component from
    one subject influences the same component in another subject in the future.
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        parameters: ModuleParameters,
        num_subjects: int,
        dimension_size: int,
        self_dependent: bool,
        dimension_names: Optional[List[str]] = None,
        coordination_samples: Optional[ModuleSamples] = None,
        coordination_random_variable: Optional[pm.Distribution] = None,
        latent_component_random_variable: Optional[pm.Distribution] = None,
        time_steps_in_coordination_scale: Optional[np.array] = None,
        observed_values: Optional[TensorTypes] = None,
        common_cause: bool = False,
        common_cause_samples: Optional[ModuleSamples] = None,
        common_cause_random_variable: Optional[pm.Distribution] = None,
    ):
        """
        Creates an abstract latent component module.

        @param uuid: string uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param parameters: parameters of the latent component.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param self_dependent: whether the latent variables in the component are tied to the
            past values from the same subject. If False, coordination will blend the previous
            latent value of a different subject with the value of the component at time t = 0 for
            the current subject (the latent component's prior for that subject).
        @param dimension_names: the names of each dimension of the latent component. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        @param coordination_samples: coordination samples to be used in a call to draw_samples.
            This variable must be set before such a call.
        @param coordination_random_variable: coordination random variable to be used in a call to
            create_random_variables. This variable must be set before such a call.
        @param latent_component_random_variable: latent component random variable to be used in a
            call to create_random_variables. If not set, it will be created in such a call.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param observed_values: observations for the latent component random variable. If a value
            is set, the variable is not latent anymore.
        @param common_cause: whether to use a common cause chain or not.
        @param common_cause_samples: optional common cause samples to be used in a call to
            draw_samples. This variable must be set before such a call if common cause is used.
        @param common_cause_random_variable: an optional common cause random variable.
        @raise ValueError: if the number of elements in dimension_names do not match the
            dimension_size.
        """

        if dimension_names is not None and len(dimension_names) != dimension_size:
            raise ValueError(
                f"The number of items in dimension_names ({len(dimension_names)}) must match the "
                f"dimension_size ({dimension_size})."
            )

        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=parameters,
            observed_values=observed_values,
        )

        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.self_dependent = self_dependent
        self.dimension_names = dimension_names
        self.coordination_samples = coordination_samples
        self.coordination_random_variable = coordination_random_variable
        self.latent_component_random_variable = latent_component_random_variable
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.common_cause = common_cause
        self.common_cause_samples = common_cause_samples
        self.common_cause_random_variable = common_cause_random_variable

    @property
    def dimension_coordinates(self) -> Union[List[str], np.ndarray]:
        """
        Gets a list of values representing the names of each dimension.

        @return: a list of dimension names.
        """
        return (
            np.arange(self.dimension_size)
            if self.dimension_names is None
            else self.dimension_names
        )

    @abstractmethod
    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Draws latent component samples using ancestral sampling and some blending strategy with
        coordination and different subjects. This method must be implemented by concrete
        subclasses.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if coordination_samples is None.
        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if self.coordination_samples is None:
            raise ValueError(
                "No coordination samples. Please set coordination_samples "
                "before invoking the draw_samples method."
            )

    @abstractmethod
    def create_random_variables(self):
        """
        Creates parameters and latent component variables in a PyMC model.

        @raise ValueError: if coordination_random_variable is None.
        """
        super().create_random_variables()

        if self.coordination_random_variable is None:
            raise ValueError(
                "Coordination variable is undefined. Please set "
                "coordination_random_variable before invoking the "
                "create_random_variables method."
            )

        self._add_coordinates()

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model.
        """

        if self.dimension_size > 0:
            self.pymc_model.add_coord(
                name=self.dimension_axis_name, values=self.dimension_coordinates
            )


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class LatentComponentSamples(ModuleSamples):
    """
    This class stores samples generated by a latent component.
    """

    def __init__(
        self,
        values: Union[List[np.ndarray], np.ndarray],
        time_steps_in_coordination_scale: Union[List[np.ndarray], np.ndarray],
    ):
        """
        Creates an object to store samples.

        @param values: sampled values of the latent component. For serial components, this will be
        a list of time series of values of different sizes. For non-serial components, this will be
        a tensor as the number of observations in time do not change for different sampled time
        series.
        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different time scale from the time scale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the latent component. For serial components, this will be a list of time series of
        indices of different sizes. For non-serial components, this will be a tensor as the number
        of observations in time do not change for different sampled time series.
        """
        super().__init__(values=values)

        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
