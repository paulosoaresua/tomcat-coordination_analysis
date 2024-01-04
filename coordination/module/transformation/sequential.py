from __future__ import annotations

from typing import List, Optional

import pymc as pm

from coordination.module.module import ModuleSamples
from coordination.module.transformation.transformation import Transformation


class Sequential(Transformation):
    """
    This class represents a transformation that executes other transformations in a sequential
    manner.
    """

    def __init__(
        self,
        child_transformations: List[Transformation],
        input_samples: Optional[ModuleSamples] = None,
        input_random_variable: Optional[pm.Distribution] = None,
        output_random_variable: Optional[pm.Distribution] = None,
    ):
        """
        Creates a sequential transformation.

        @param child_transformations: transformations to be executed in sequence.
        @param input_samples: samples transformed in a call to draw_samples. This variable must be
            set before such a call.
        @param input_random_variable: random variable to be transformed in a call to
            create_random_variables. This variable must be set before such a call.
        @param output_random_variable: transformed random variable. If set, not transformation is
            performed in a call to create_random_variables.
        """
        super().__init__(
            uuid=None,
            pymc_model=None,
            parameters=None,
            input_samples=input_samples,
            input_random_variable=input_random_variable,
            output_random_variable=output_random_variable,
            observed_values=None,
            axis=None,
        )

        self.child_transformations = child_transformations

    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Transforms input samples by applying different transformations in sequence.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: transformed samples for each series.
        """
        super().draw_samples(seed, num_series)

        transformed_samples = None
        self.child_transformations[0].input_samples = self.input_samples
        for transformation in self.child_transformations:
            if transformed_samples is not None:
                transformation.input_samples = transformed_samples

            transformed_samples = transformation.draw_samples(seed, num_series)

        return transformed_samples

    def create_random_variables(self):
        """
        Creates an output random variable by applying different transformations in sequence.
        """
        super().create_random_variables()

        transformed_random_variable = None
        self.child_transformations[0].input_random_variable = self.input_random_variable
        for transformation in self.child_transformations:
            if transformed_random_variable is not None:
                transformation.input_random_variable = transformed_random_variable

            transformation.create_random_variables()
            transformed_random_variable = transformation.output_random_variable

        self.output_random_variable = self.child_transformations[
            -1
        ].output_random_variable
