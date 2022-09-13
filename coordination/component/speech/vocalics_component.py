from __future__ import annotations
from typing import Any, List, Optional, Tuple

from datetime import datetime
from enum import Enum
import json
import logging
import numpy as np
import os
import pickle

from coordination.common.sparse_series import SparseSeries
from coordination.entity.vocalics import Utterance, Vocalics
from coordination.entity.vocalics_series import VocalicsSeries
from coordination.plot.vocalics import plot_vocalic_features, plot_utterance_durations

UTTERANCE_MISSING_VOCALICS_DURATION_THRESHOLD = 1

logger = logging.getLogger()


class VocalicsSparseSeries(SparseSeries):
    def __init__(self, utterances: List[Optional[SegmentedUtterance]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.utterances = utterances


class SegmentedUtterance:
    def __init__(self, subject_id: str, start: datetime, end: datetime, text: str):
        self.subject_id = subject_id
        self.start = start
        self.end = end
        self.text = text

        self.vocalic_series: VocalicsSeries = VocalicsSeries(np.array([]), [])

    @classmethod
    def from_utterance(cls, utterance: Utterance, start: datetime, end: datetime) -> SegmentedUtterance:
        return cls(utterance.subject_id, start, end, utterance.text)


class SegmentationMethod(Enum):
    # If the current utterance overlaps with the next, it ends the current utterance when the next one starts
    TRUNCATE_CURRENT = 1

    # If the current utterance overlaps with the next, it starts the next utterance when the current one finishes.
    # The next utterance will be discarded if the current utterance finishes after the next one.
    TRUNCATE_NEXT = 2

    # It considers the entire current and next utterances regardless of whether they overlap.
    KEEP_ALL = 3


class VocalicsComponent:
    def __init__(self,
                 series_a: List[SegmentedUtterance],
                 series_b: List[SegmentedUtterance],
                 feature_names: List[str]):
        # One series per subject in turn.
        self.series_a = series_a
        self.series_b = series_b
        self.features = feature_names

    @classmethod
    def from_vocalics(cls, vocalics: Vocalics,
                      segmentation_method: SegmentationMethod = SegmentationMethod.TRUNCATE_CURRENT):
        if segmentation_method == SegmentationMethod.TRUNCATE_CURRENT:
            return VocalicsComponent._split_with_current_utterance_truncation(vocalics)
        elif segmentation_method == SegmentationMethod.TRUNCATE_NEXT:
            return VocalicsComponent._split_with_next_utterance_truncation(vocalics)
        elif segmentation_method == SegmentationMethod.KEEP_ALL:
            return VocalicsComponent._split_with_overlap(vocalics)
        else:
            raise Exception(f"{segmentation_method} is an invalid segmentation method.")

    @classmethod
    def from_trial_directory(cls, trial_dir: str) -> VocalicsComponent:
        vocalics_component_a_path = f"{trial_dir}/vocalics_component_a.pkl"
        vocalics_component_b_path = f"{trial_dir}/vocalics_component_b.pkl"
        feature_names_path = f"{trial_dir}/features.txt"

        if not os.path.exists(vocalics_component_a_path):
            raise Exception(f"Could not find the file vocalics_component_a.pkl in {trial_dir}.")

        if not os.path.exists(vocalics_component_b_path):
            raise Exception(f"Could not find the file vocalics_component_b.pkl in {trial_dir}.")

        if not os.path.exists(feature_names_path):
            raise Exception(f"Could not find the file features.txt in {trial_dir}.")

        with open(vocalics_component_a_path, "rb") as f:
            series_a = pickle.load(f)

        with open(vocalics_component_b_path, "rb") as f:
            series_b = pickle.load(f)

        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)

        return cls(series_a, series_b, feature_names)

    def sparse_series(self, num_time_steps: int) -> Tuple[VocalicsSparseSeries, VocalicsSparseSeries]:
        def series_to_seconds(utterances: List[SegmentedUtterance],
                              initial_timestamp: datetime) -> VocalicsSparseSeries:
            values = np.zeros((utterances[0].vocalic_series.num_series, num_time_steps))
            mask = np.zeros(num_time_steps)  # 1 for time steps with observation, 0 otherwise
            timestamps: List[Optional[datetime]] = [None] * num_time_steps
            segmented_utterances: List[Optional[SegmentedUtterance]] = [None] * num_time_steps

            for i, utterance in enumerate(utterances):
                # We consider that the observation is available at the end of an utterance. We take the average vocalics
                # per feature within the utterance as a measurement at the respective time step.
                time_step = int((utterance.end - initial_timestamp).total_seconds())
                if time_step >= num_time_steps:
                    logger.warning(f"""Time step {time_step} exceeds the number of time steps {num_time_steps} at 
                                   utterance {i} out of {len(utterances)} ending at {utterance.end.isoformat()} 
                                   considering an initial timestamp of {initial_timestamp.isoformat()}.""")
                    break

                values[:, time_step] = utterance.vocalic_series.values.mean(axis=1)
                mask[time_step] = 1
                timestamps[time_step] = utterance.end
                segmented_utterances[time_step] = utterance

            return VocalicsSparseSeries(values=values, mask=mask, timestamps=timestamps,
                                        utterances=segmented_utterances)

        # The first utterance always goes in series A
        earliest_timestamp = self.series_a[0].start
        sparse_series_a = series_to_seconds(self.series_a, earliest_timestamp)
        sparse_series_b = series_to_seconds(self.series_b, earliest_timestamp)

        return sparse_series_a, sparse_series_b

    def plot_features(self, axs: List[Any], num_time_steps: int, timestamp_as_index: bool = True,
                      normalize: bool = False):
        sparse_series_a, sparse_series_b = self.sparse_series(num_time_steps)
        if normalize:
            sparse_series_a.normalize()
            sparse_series_b.normalize()
        plot_vocalic_features(axs, sparse_series_a, sparse_series_b, self.features, timestamp_as_index)

    def plot_utterance_durations(self, ax: Any):
        utterances_per_subject = {
            "Series A": self.series_a,
            "Series B": self.series_b
        }

        return plot_utterance_durations(ax, utterances_per_subject)

    @classmethod
    def _split_with_current_utterance_truncation(cls, vocalics: Vocalics) -> VocalicsComponent:
        """
        Split utterances from different individuals into two series. Merging utterances of the current speaker in the
        active series, and changing series when another person is the speaker. If speeches overlap, it truncates the
        utterance and associated vocalics of the current speaker.
        """
        # Single list with the utterances from all the subjects
        utterances: List[Utterance] = []
        for u in vocalics.utterances_per_subject.values():
            utterances.extend(u)

        # Remove utterances with no vocalics
        utterances = VocalicsComponent._filter_utterances(utterances)

        # We sort the utterances by timestamp, regardless of the subject they belong to.
        utterances.sort(key=lambda utterance: utterance.start)

        series_a: List[SegmentedUtterance] = []
        series_b: List[SegmentedUtterance] = []

        series_a_active = True
        segmented_vocalic_values = np.array([])
        segmented_vocalic_timestamps: List[datetime] = []
        segmented_utterance = None
        for i, current_utterance in enumerate(utterances):

            # Start a new segmented utterance if the previous one is completed or not available
            if segmented_utterance is None:
                segmented_utterance = SegmentedUtterance.from_utterance(current_utterance, current_utterance.start,
                                                                        current_utterance.end)

            next_utterance = utterances[i + 1] if i < len(utterances) - 1 else None

            for j in range(current_utterance.vocalic_series.size):
                # If vocalics of current utterance overlaps with the next, then move on to next utterance
                if next_utterance is not None and current_utterance.vocalic_series.timestamps[j] > next_utterance.start:
                    segmented_utterance.end = next_utterance.start
                    break

                # Add vocalics to the current segment
                if len(segmented_vocalic_values) == 0:
                    segmented_vocalic_values = current_utterance.vocalic_series.values[:, j, np.newaxis]
                else:
                    segmented_vocalic_values = np.concatenate(
                        [segmented_vocalic_values, current_utterance.vocalic_series.values[:, j, np.newaxis]], axis=1)
                segmented_vocalic_timestamps.append(current_utterance.vocalic_series.timestamps[j])

            # If the subject of the next utterance is different, we start a new segment.
            if next_utterance is None or next_utterance.subject_id != current_utterance.subject_id:
                segmented_vocalic_series = VocalicsSeries(segmented_vocalic_values, segmented_vocalic_timestamps)
                segmented_utterance.vocalic_series = segmented_vocalic_series

                if series_a_active:
                    series_a.append(segmented_utterance)
                else:
                    series_b.append(segmented_utterance)

                segmented_utterance = None
                segmented_vocalic_values = np.array([])
                segmented_vocalic_timestamps: List[datetime] = []
                series_a_active = not series_a_active
            elif next_utterance is not None and next_utterance.subject_id == current_utterance.subject_id:
                segmented_utterance.end = next_utterance.end

        return cls(series_a, series_b, vocalics.features)

    @classmethod
    def _split_with_next_utterance_truncation(cls, vocalics: Vocalics) -> VocalicsComponent:
        raise RuntimeError("Not implemented")

    @classmethod
    def _split_with_overlap(cls, vocalics: Vocalics) -> VocalicsComponent:
        raise RuntimeError("Not implemented")

    @staticmethod
    def _filter_utterances(utterances: List[Utterance]) -> List[Utterance]:
        """
        Remove utterances with no vocalics.
        """

        utterance_missing_vocalics_short_duration = []
        utterance_missing_vocalics_long_duration = []

        def filter_criteria(utterance: Utterance):
            # We filter out utterances with no vocalics. We keep a list of the utterances to write to the log later.
            if utterance.vocalic_series.size == 0:
                if utterance.duration_in_seconds < UTTERANCE_MISSING_VOCALICS_DURATION_THRESHOLD:
                    utterance_missing_vocalics_short_duration.append(utterance)
                else:
                    utterance_missing_vocalics_long_duration.append(utterance)

                return False

            return True

        utterances = list(filter(filter_criteria, utterances))

        # Log utterances with no vocalics
        for utterance in utterance_missing_vocalics_short_duration:
            logger.warning(f"""Utterance starting at {utterance.start.isoformat()} and ending 
                at {utterance.end.isoformat()} is short and does not have any vocalics. Text: {utterance.text}""")

        for utterance in utterance_missing_vocalics_long_duration:
            logger.warning(f"""Utterance starting at {utterance.start.isoformat()} and ending 
                at {utterance.end.isoformat()} is long and does not have any vocalics. Text: {utterance.text}""")

        return utterances

    def save(self, out_dir: str, save_feature_names: bool = True):
        with open(f"{out_dir}/vocalics_component_a.pkl", "wb") as f:
            pickle.dump(self.series_a, f)

        with open(f"{out_dir}/vocalics_component_b.pkl", "wb") as f:
            pickle.dump(self.series_b, f)

        if save_feature_names:
            with open(f"{out_dir}/features.txt", "w") as f:
                json.dump(self.features, f)
