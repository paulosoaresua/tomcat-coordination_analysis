from __future__ import annotations

from abc import ABC
from typing import Optional

import pymc as pm

from coordination.module.latent_component.latent_component import \
    LatentComponent
from coordination.module.module import ModuleSamples
import logging


class NullLatentComponent(LatentComponent, ABC):
    """
    This class represents a null latent system component. It works only as a container to
    coordination data that is accessed by observations that bypass latent components. For instance,
    spike observations.
    """

    def __init__(
        self,
        coordination_samples: Optional[ModuleSamples] = None,
        coordination_random_variable: Optional[pm.Distribution] = None,
    ):
        """
        Creates a null latent component module.

        @param coordination_samples: coordination samples to be used in a call to draw_samples.
            This variable must be set before such a call.
        @param coordination_random_variable: coordination random variable to be used in a call to
            create_random_variables. This variable must be set before such a call.
        """

        super().__init__(
            uuid="null",
            pymc_model=None,
            parameters=None,
            num_subjects=0,
            dimension_size=0,
            self_dependent=False,
            coordination_samples=coordination_samples,
            coordination_random_variable=coordination_random_variable,
        )

    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Calls superclass method to ensure coordination samples were provided.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

    def create_random_variables(self):
        """
        Calls superclass method to ensure coordination random variable was provided.
        """
        super().create_random_variables()

        logging.info(f"Fitting {self.__class__.__name__}.")
