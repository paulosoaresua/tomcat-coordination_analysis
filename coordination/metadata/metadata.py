from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class Metadata:
    """
    This class represents a generic metadata.
    """

    def __init__(
            self,
            time_steps_in_coordination_scale: np.array,
            observed_values: np.ndarray,
            normalization_method: str):
        """
        Creates a serial metadata:

        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param observed values for the serial component.
        @param normalization_method: normalization method to apply on observations.
        """
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.observed_values = observed_values
        self.normalization_method = normalization_method

    @abstractmethod
    def truncate(self, max_time_step: int) -> Metadata:
        """
        Gets a new metadata making sure time steps do not surpass a maximum value provided.

        @param max_time_step: maximum time step value.
        @return: new metadata with adjusted arrays.
        """
        pass

    @property
    def normalized_observations(self) -> np.ndarray:
        """
        Normalize observations with some method.

        @return normalized observations.
        """
        return self.normalize(self.observed_values)

    @abstractmethod
    def normalize(self, observations: np.ndarray, time_interval: Optional[Tuple[int, int]] = None):
        """
        Normalize observations with some method.

        @param observations: observations to be normalized.
        @param time_interval: optional time interval. If provided, only the portion of data
            determined by the interval will be normalized and returned.
        @return normalized observations.
        """
        pass

    @abstractmethod
    def split_observations_per_subject(
            self,
            observations: np.ndarray,
            normalize: bool,
            skip_first: Optional[int] = None,
            skip_last: Optional[int] = None) -> List[np.ndarray]:
        """
        Returns a list of observations per speaker as a list of arrays.

        @param observations: observations to be split.
        @param normalize: whether observations must be normalized before retrieved.
        @return observations per subjects.
        @param skip_first: number of time steps to skip.
        @param skip_last: number of time steps to not to include.
        """
        pass
