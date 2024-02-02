from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np
import pymc as pm

from coordination.module.module import ModuleSamples
from coordination.module.transformation.transformation import Transformation


class DimensionReduction(Transformation):
    """
    This class represents a transformation that generates an output from selected dimensions of
    the input.
    """

    def __init__(
        self,
        keep_dimensions: List[int],
        input_samples: Optional[ModuleSamples] = None,
        input_random_variable: Optional[pm.Distribution] = None,
        output_random_variable: Optional[pm.Distribution] = None,
        axis: int = 0,
    ):
        """
        Creates a dimension reduction transformation.

        @param keep_dimensions: list of dimensions to keep in the output.
        @param input_samples: samples transformed in a call to draw_samples. This variable must be
            set before such a call.
        @param input_random_variable: random variable to be transformed in a call to
            create_random_variables. This variable must be set before such a call.
        @param output_random_variable: transformed random variable. If set, not transformation is
            performed in a call to create_random_variables.
        @param axis: axis to apply the transformation.
        """
        super().__init__(
            uuid=None,
            pymc_model=None,
            parameters=None,
            input_samples=input_samples,
            input_random_variable=input_random_variable,
            output_random_variable=output_random_variable,
            observed_values=None,
            axis=axis,
        )

        self.keep_dimensions = keep_dimensions

    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Transforms input samples by picking selected dimensions.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: transformed samples for each series.
        """
        super().draw_samples(seed, num_series)

        transformed_samples_values = []
        for i, sampled_series in enumerate(self.input_samples.values):
            transformed_samples_values.append(
                np.take(sampled_series, self.keep_dimensions, axis=self.axis)
            )

        transformed_samples = deepcopy(self.input_samples)
        transformed_samples.values = (
            np.array(transformed_samples_values)
            if isinstance(self.input_samples.values, np.ndarray)
            else transformed_samples_values
        )
        return transformed_samples

    def create_random_variables(self):
        """
        Creates an output random variable by selecting the chosen dimensions in the input random
        variable.
        """
        super().create_random_variables()

        self.output_random_variable = self.input_random_variable.take(
            self.keep_dimensions, axis=self.axis
        )
