from __future__ import annotations
import numpy as np
from coordination.metadata.metadata import Metadata
from dataclasses import dataclass
from coordination.common.normalization import (NORMALIZATION_PER_SUBJECT_AND_FEATURE,
                                               NORMALIZATION_PER_FEATURE,
                                               normalize_serialized_data_per_feature,
                                               normalize_serialized_data_per_subject_and_feature)
from copy import deepcopy


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
        super().__init__(time_steps_in_coordination_scale, observed_values, normalization_method)

        self.num_subjects = num_subjects
        self.subject_indices = subject_indices
        self.prev_time_same_subject = prev_time_same_subject
        self.prev_time_diff_subject = prev_time_diff_subject

    def truncate(self, max_time_step: int) -> SerialMetadata:
        """
        Gets a new metadata making sure time steps do not surpass a maximum value provided.

        @param max_time_step: maximum time step value.
        @return: new metadata with adjusted arrays.
        """
        ts = self.time_steps_in_coordination_scale
        if ts is None:
            return deepcopy(self)

        ts = ts[ts < max_time_step]
        return SerialMetadata(
            num_subjects=self.num_subjects,
            time_steps_in_coordination_scale=ts,
            subject_indices=self.subject_indices[:len(ts)],
            prev_time_same_subject=self.prev_time_same_subject[:len(ts)],
            prev_time_diff_subject=self.prev_time_diff_subject[:len(ts)],
            observed_values=self.observed_values[...,
                            :len(ts)] if self.observed_values is not None else None,
            normalization_method=self.normalization_method
        )

    def normalize(self, observations: np.ndarray, time_interval: Optional[Tuple[int, int]] = None):
        """
        Normalize observations with some method.

        @param observations: observations to be normalized.
        @param time_interval: optional time interval. If provided, only the portion of data
            determined by the interval will be normalized and returned.
        @return normalized observations.
        """
        if time_interval is None:
            obs = observations
            sub_idx = self.subject_indices
        else:
            obs = observations[..., time_interval[0]:time_interval[1]]
            sub_idx = self.subject_indices[time_interval[0]:time_interval[1]]

        if self.normalization_method is None or observations is None:
            return obs

        if self.normalization_method == NORMALIZATION_PER_FEATURE:
            return normalize_serialized_data_per_feature(obs)

        if self.normalization_method == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            return normalize_serialized_data_per_subject_and_feature(
                data=obs,
                subject_indices=sub_idx,
                num_subjects=self.num_subjects)

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

        result = []
        lb = 0 if skip_first is None else skip_first
        ub = obs.shape[-1] if skip_last is None else -skip_last
        for s in range(self.num_subjects):
            idx = [t for t, subject in enumerate(self.subject_indices[lb:ub]) if s == subject]
            result.append(obs[..., lb:ub][..., idx])

        return result
