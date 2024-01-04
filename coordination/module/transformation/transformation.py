from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.module import Module, ModuleParameters, ModuleSamples


class Transformation(ABC, Module):
    """
    This class represents a transformation that can be applied to variables created by other
    modules.
    """

    def __init__(
        self,
        uuid: str,
        pymc_model: pm.Model,
        parameters: ModuleParameters,
        input_samples: Optional[ModuleSamples] = None,
        input_random_variable: Optional[pm.Distribution] = None,
        output_random_variable: Optional[pm.Distribution] = None,
        axis: int = 0,
        observed_values: Optional[Union[TensorTypes, Dict[str, TensorTypes]]] = None,
    ):
        """
        Creates a transformation.

        @param uuid: string uniquely identifying the transformation in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param parameters: parameters of the module.
        @param input_samples: samples transformed in a call to draw_samples. This variable must be
            set before such a call.
        @param input_random_variable: random variable to be transformed in a call to
            create_random_variables. This variable must be set before such a call.
        @param output_random_variable: transformed random variable. If set, not transformation is
            performed in a call to create_random_variables.
        @param observed_values: observations for the weights random variable. If a value
            is set, the variable is not latent anymore.
        @param axis: axis to apply the transformation.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=parameters,
            observed_values=observed_values,
        )

        self.input_samples = input_samples
        self.input_random_variable = input_random_variable
        self.output_random_variable = output_random_variable
        self.axis = axis

    @abstractmethod
    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Transforms input samples.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @raise ValueError: if input_samples is None.
        @return: transformed samples for each series.
        """
        super().draw_samples(seed, num_series)

        if self.input_samples is None:
            raise ValueError(
                "No input samples. Please set input_samples before invoking the "
                "draw_samples method."
            )

    @abstractmethod
    def create_random_variables(self):
        """
        Creates parameters and transformation variables in a PyMC model.
        The transformation is applied here.

        @raise ValueError: if coordination_random_variable is None.
        """
        super().create_random_variables()

        if self.input_random_variable is None:
            raise ValueError(
                "Input variable is undefined. Please set "
                "input_random_variable before invoking the create_random_variables "
                "method."
            )
