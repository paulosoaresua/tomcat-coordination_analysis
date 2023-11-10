from __future__ import annotations

from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.parametrization2 import Parameter
from coordination.common.utils import set_random_seed


class Module:
    """
    This class represents a generic module of a coordination model.
    """

    def __init__(self,
                 uuid: str,
                 pymc_model: pm.Model,
                 parameters: Optional[ModuleParameters],
                 observed_values: Optional[TensorTypes]):
        """

        @param uuid: unique identifier of the module.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param parameters: parameters of the module.
        @param observed_values: observations for the non-parameter random variable in the model.
            Values for the parameter variables are set directly in the module's parameters
            attribute. If a value is set, the variable is not latent anymore.
        """
        self.uuid = uuid
        self.pymc_model = pymc_model
        self.parameters = parameters
        self.observed_values = observed_values

    @property
    def time_axis_name(self) -> str:
        return f"{self.uuid}_time"

    @property
    def dimension_axis_name(self) -> str:
        return f"{self.uuid}_dimension"

    @property
    def subject_axis_name(self) -> str:
        return f"{self.uuid}_subject"

    @property
    def parameter_names(self) -> List[str]:
        """
        Gets the names of all the parameters used in the distributions of a latent variables
        recursively.

        @return: a list with the parameter names.
        """

        parameter_names = []

        if self.parameters:
            parameter_names.extend(self.parameters.parameter_names)

        # Look for other modules nested to this one and gather parameters from them.
        attributes = vars(self)
        for _, attribute in attributes.items():
            if isinstance(attribute, Module):
                parameter_names.extend(attribute.parameter_names)
            elif isinstance(attribute, List):
                for list_item in attribute:
                    if isinstance(list_item, Module):
                        parameter_names.extend(list_item.parameter_names)

        return parameter_names

    @property
    def random_variables(self) -> Dict[str, pm.Distribution]:
        """
        Gets all random variables created in the module as a dictionary indexed by their names.

        @return: dictionary of random variables defined in the module.
        """

        rv_dict = {}
        for rv in self.pymc_model.basic_RVs:
            rv_dict[rv.name] = rv

        return rv_dict

    def clear_parameter_values(self):
        """
        Clears the values of all the parameters recursively through the nested modules. The
        parameter hyper-priors are preserved.
        """
        if self.parameters:
            self.parameters.clear_values()

        # Look for other modules nested to this one and clear parameters from them.
        attributes = vars(self)
        for _, attribute in attributes.items():
            if isinstance(attribute, Module):
                attribute.clear_parameter_values()
            elif isinstance(attribute, List):
                for list_item in attribute:
                    if isinstance(list_item, Module):
                        list_item.clear_parameter_values()

        return parameter_names

    @abstractmethod
    def draw_samples(self,
                     seed: Optional[int],
                     num_series: int) -> ModuleSamples:
        """
        Draws samples using ancestral sampling.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        """
        set_random_seed(seed)

    @abstractmethod
    def create_random_variables(self):
        """
        Creates random variables in a PyMC model.
        """
        pass


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class ModuleParameters(ABC):
    """
    This class stores values and hyper-priors of the parameters of a module.
    """

    def clear_values(self):
        """
        Set values of the parameters to None. Parameters with None value will be fit to the data
        along with other latent values in the model.
        """
        attributes = vars(self)
        for _, parameter in attributes.items():
            if isinstance(parameter, Parameter):
                parameter.value = None

    @property
    def parameter_names(self) -> List[str]:
        """
        Gets a list of unique names of the distribution parameters in the module.

        @return: list of parameter names.
        """
        names = []
        attributes = vars(self)
        for _, parameter in attributes.items():
            if isinstance(parameter, Parameter):
                names.append(parameter.uuid)

        return names


class ModuleSamples(ABC):
    """
    This class stores samples generated by a module.
    """

    def __init__(self,
                 values: Optional[Union[List[np.ndarray], np.ndarray]]):
        """
        Creates an object to store samples.

        @param values: sampled values of a module. For serial modules, this will be
            a list of time series of values of different sizes. For non-serial modules, this
            will be a tensor as the number of observations in time do not change for different
            sampled time series.
        """

        self.values = values

    @property
    def num_time_steps(self) -> Union[int, np.array]:
        """
        Gets the number of time steps
        @return: number of time steps.
        """

        if isinstance(self.values, List):
            # For a list of sampled series, they can have a different number of time steps. If
            # a scalar is returned, otherwise an array is returned with the number of time steps in
            # each individual series.
            sizes = np.array([sampled_series.shape[-1] for sampled_series in self.values])
            if len(sizes) == 0:
                return 0
            elif len(sizes) == 1:
                return sizes[0]
            else:
                return sizes
        else:
            return self.values.shape[-1]

    @property
    def num_series(self) -> int:
        """
        Gets the number of series.

        @return: number of series.
        """

        if isinstance(self.values, List):
            return len(self.values)
        else:
            return self.values.shape[0]
