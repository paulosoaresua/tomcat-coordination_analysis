from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metadata:
    """
    This class represents a generic metadata.
    """

    @abstractmethod
    def truncate(self, max_time_step: int) -> Metadata:
        """
        Gets a new metadata making sure time steps do not surpass a maximum value provided.

        @param max_time_step: maximum time step value.
        @return: new metadata with adjusted arrays.
        """
        pass

    @property
    @abstractmethod
    def normalized_observations(self) -> np.ndarray:
        """
        Normalize observations with some method.

        @return normalized observations.
        """
        pass
