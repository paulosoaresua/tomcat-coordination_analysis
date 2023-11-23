from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.parametrization2 import (Parameter,
                                                  HalfNormalParameterPrior)
from coordination.module.module import Module, ModuleParameters, ModuleSamples


class Observation(ABC, Module):
    """
    This class represents an observation (O) from a latent system component (A). Observations are
    evidence to the model. This implementation samples observation from a Gaussian distribution
     centered on some transformation, g(.),  of the latent components, i.e., O ~ N(g(A), var_o).
    """

    def __init__(self,
                 uuid: str,
                 pymc_model: pm.Model,
                 parameters: ModuleParameters,
                 num_subjects: int,
                 dimension_size: int,
                 dimension_names: Optional[List[str]] = None,
                 coordination_samples: Optional[ModuleSamples] = None,
                 latent_component_samples: Optional[ModuleSamples] = None,
                 coordination_random_variable: Optional[pm.Distribution] = None,
                 latent_component_random_variable: Optional[pm.Distribution] = None,
                 observation_random_variable: Optional[pm.Distribution] = None,
                 observed_values: Union[TensorTypes, Dict[str, TensorTypes]] = None):
        """
        Creates an observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param parameters: parameters of the module.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        @param coordination_samples: coordination samples.
        @param latent_component_samples: latent component samples.
        @param coordination_random_variable: coordination random variable.
        @param latent_component_random_variable: latent component random variable.
        @param observation_random_variable: observation random variable to be used in a
            call to create_random_variables. If not set, it will be created in such a call.
        @raise ValueError: if the number of elements in dimension_names do not match the
            dimension_size.
        """

        if dimension_names is not None and len(dimension_names) != dimension_size:
            raise ValueError(
                f"The number of items in dimension_names ({len(dimension_names)}) must match the "
                f"dimension_size ({dimension_size}).")

        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=parameters,
            observed_values=observed_values)

        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.dimension_names = np.arange(
            dimension_size) if dimension_names is None else dimension_names
        self.coordination_samples = coordination_samples
        self.latent_component_samples = latent_component_samples
        self.coordination_random_variable = coordination_random_variable
        self.latent_component_random_variable = latent_component_random_variable
        self.observation_random_variable = observation_random_variable

    @property
    def dimension_coordinates(self) -> Union[List[str], np.ndarray]:
        """
        Gets a list of values representing the names of each dimension.

        @return: a list of dimension names.
        """
        return np.arange(
            self.dimension_size) if self.dimension_names is None else self.dimension_names

    @abstractmethod
    def create_random_variables(self):
        """
        Updates model coordinates. The creation of random variables and parameters must be done by
        the subclasses.
        """
        super().create_random_variables()
        self._add_coordinates()

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model.
        """
        self.pymc_model.add_coord(name=self.dimension_axis_name,
                                  values=self.dimension_coordinates)
