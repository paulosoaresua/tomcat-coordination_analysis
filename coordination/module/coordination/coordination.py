from __future__ import annotations

from abc import ABC
from typing import Optional

import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.module import Module, ModuleParameters


class Coordination(ABC, Module):
    UUID = "coordination"

    def __init__(
        self,
        pymc_model: pm.Model,
        parameters: ModuleParameters,
        num_time_steps: int,
        coordination_random_variable: Optional[pm.Distribution] = None,
        observed_values: Optional[TensorTypes] = None,
    ):
        """
        Creates a coordination module.

        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param parameters: parameters of the module.
        @param num_time_steps: number of time steps in the coordination scale.
        @param coordination_random_variable: random variable to be used in a call to
            create_random_variables. If not set, it will be created in such a call.
        @param observed_values: observations for the coordination random variable. If a value
            is set, the variable is not latent anymore.
        """
        super().__init__(
            uuid=Coordination.UUID,
            pymc_model=pymc_model,
            parameters=parameters,
            observed_values=observed_values,
        )

        self.num_time_steps = num_time_steps
        self.coordination_random_variable = coordination_random_variable
