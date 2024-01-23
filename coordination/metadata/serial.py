from __future__ import annotations
import numpy as np
from coordination.metadata.metadata import Metadata
from dataclasses import dataclass
from coordination.common.normalization import (NORMALIZATION_PER_SUBJECT_AND_FEATURE,
                                               NORMALIZATION_PER_FEATURE,
                                               normalize_serialized_data_per_feature,
                                               normalize_serialized_data_per_subject_and_feature)


@dataclass
class SerialMetadata(Metadata):
    """
    This class holds metadata for serial modules.
    """

    def __init__(
            self,
            num_subjects: int,
            time_steps_in_coordination_scale: np.array,
            subject_indices: np.ndarray,
            prev_time_same_subject: np.ndarray,
            prev_time_diff_subject: np.ndarray,
            observed_values: np.ndarray,
            normalization_method: str):
        """
        Creates a serial metadata:

        @param num_subjects: number of subjects.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param subject_indices: array of numbers indicating which subject is associated to the
            latent component at every time step (e.g. the current speaker for a speech component).
            In serial components, only one user's latent component is observed at a time. This
            array indicates which user that is. This array contains no gaps. The size of the array
            is the number of observed latent component in time, i.e., latent component time
            indices with an associated subject.
        @param prev_time_same_subject: time indices indicating the previous observation of the
            latent component produced by the same subject at a given time. For instance, the last
            time when the current speaker talked. This variable must be set before a call to
            update_pymc_model.
        @param prev_time_diff_subject: similar to the above but it indicates the most recent time
            when the latent component was observed for a different subject. This variable must be
            set before a call to update_pymc_model.
        @param observed values for the serial component.
        @param normalization_method: normalization method to apply on observations.
        """
        self.num_subjects = num_subjects
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.subject_indices = subject_indices
        self.prev_time_same_subject = prev_time_same_subject
        self.prev_time_diff_subject = prev_time_diff_subject
        self.observed_values = observed_values
        self.normalization_method = normalization_method

    def truncate(self, max_time_step: int) -> SerialMetadata:
        """
        Gets a new metadata making sure time steps do not surpass a maximum value provided.

        @param max_time_step: maximum time step value.
        @return: new metadata with adjusted arrays.
        """
        ts = self.time_steps_in_coordination_scale
        ts = ts[ts < max_time_step]

        return SerialMetadata(
            num_subjects=self.num_subjects,
            time_steps_in_coordination_scale=ts,
            subject_indices=self.subject_indices[:len(ts)],
            prev_time_same_subject=self.prev_time_same_subject[:len(ts)],
            prev_time_diff_subject=self.prev_time_diff_subject[:len(ts)],
            observed_values=self.observed_values[..., :len(ts)],
            normalization_method=self.normalization_method
        )

    @property
    def normalized_observations(self) -> np.ndarray:
        """
        Normalize observations with some method.

        @return normalized observations.
        """
        if self.normalization_method is None:
            return self.observed_values

        if self.normalization_method == NORMALIZATION_PER_FEATURE:
            return normalize_serialized_data_per_feature(self.observed_values)

        if self.normalization_method == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            return normalize_serialized_data_per_subject_and_feature(
                data=self.observed_values,
                subject_indices=self.subject_indices,
                num_subjects=self.num_subjects)

        raise ValueError(f"Normalization ({method}) is invalid.")
