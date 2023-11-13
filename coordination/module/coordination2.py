from __future__ import annotations
from typing import Any, Optional
from abc import ABC

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.functions import sigmoid
from coordination.module.parametrization2 import Parameter, HalfNormalParameterPrior, \
    NormalParameterPrior
from coordination.common.utils import set_random_seed
from coordination.module.module import ModuleSamples, Module, ModuleParameters


class Coordination(ABC, Module):
    def __init__(self,
                 uuid: str,
                 pymc_model: pm.Model,
                 parameters: ModuleParameters,
                 num_time_steps: int,
                 coordination_random_variable: Optional[pm.Distribution] = None,
                 observed_values: Optional[TensorTypes] = None):
        """
        Creates a coordination module.

        @param uuid: string uniquely identifying the coordination module in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param parameters: parameters of the module.
        @param num_time_steps: number of time steps in the coordination scale.
        @param coordination_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param observed_values: observations for the coordination random variable. If a value
            is set, the variable is not latent anymore.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=parameters,
            observed_values=observed_values)

        self.num_time_steps = num_time_steps
        self.coordination_random_variable = coordination_random_variable
