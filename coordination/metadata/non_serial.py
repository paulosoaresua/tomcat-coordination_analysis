from __future__ import annotations
import numpy as np
from coordination.metadata.metadata import Metadata
from coordination.common.normalization import (NORMALIZATION_PER_SUBJECT_AND_FEATURE,
                                               NORMALIZATION_PER_FEATURE,
                                               normalize_non_serial_data_per_feature,
                                               normalize_non_serial_data_per_subject_and_feature)
from copy import deepcopy


class NonSerialMetadata(Metadata):
    """
    This class holds metadata for non-serial modules.
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
        super().__init__(time_steps_in_coordination_scale, observed_values, normalization_method)

    def truncate(self, max_time_step: int) -> NonSerialMetadata:
        """
        Gets a new metadata making sure time steps do not surpass a maximum value provided.

        @param max_time_step: maximum time step value.
        @return: new metadata with adjusted arrays.
        """
        ts = self.time_steps_in_coordination_scale
        if ts is None:
            return deepcopy(self)

        ts = ts[ts < max_time_step]

        return NonSerialMetadata(
            time_steps_in_coordination_scale=ts,
            observed_values=self.observed_values[...,
                            :len(ts)] if self.observed_values is not None else None,
            normalization_method=self.normalization_method
        )

    def normalize(self, observations: np.ndarray):
        """
        Normalize observations with some method.

        @param observations: observations to be normalized.
        @return normalized observations.
        """
        if self.normalization_method is None or observations is None:
            return observations

        if self.normalization_method == NORMALIZATION_PER_FEATURE:
            return normalize_non_serial_data_per_feature(observations)

        if self.normalization_method == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            return normalize_non_serial_data_per_subject_and_feature(observations)

        raise ValueError(f"Normalization ({method}) is invalid.")

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
        obs = self.normalize(observations) if normalize else observations

        if obs is None:
            return None

        lb = 0 if skip_first is None else skip_first
        ub = obs.shape[-1] if skip_last is None else -skip_last

        return obs[..., lb:ub]
