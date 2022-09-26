from __future__ import annotations
from typing import Any, Dict, List, Optional

import copy
from datetime import datetime
from enum import Enum
import json
import logging
import numpy as np
import os
import pickle

from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.entity.vocalics import Utterance, Vocalics
from coordination.plot.vocalics import plot_vocalic_features, plot_utterance_durations

UTTERANCE_MISSING_VOCALICS_DURATION_THRESHOLD = 1

logger = logging.getLogger()


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
                 segmented_utterances: List[SegmentedUtterance],
                 feature_names: List[str]):
        self.segmented_utterances = segmented_utterances
        self.features = feature_names

    @classmethod
    def from_vocalics(cls, vocalics: Vocalics,
                      segmentation_method: SegmentationMethod = SegmentationMethod.TRUNCATE_CURRENT,
                      min_vocalic_values: int = 1):
        if segmentation_method == SegmentationMethod.TRUNCATE_CURRENT:
            return VocalicsComponent._split_with_current_utterance_truncation(vocalics, min_vocalic_values)
        elif segmentation_method == SegmentationMethod.TRUNCATE_NEXT:
            return VocalicsComponent._split_with_next_utterance_truncation(vocalics, min_vocalic_values)
        elif segmentation_method == SegmentationMethod.KEEP_ALL:
            return VocalicsComponent._split_with_overlap(vocalics, min_vocalic_values)
        else:
            raise Exception(f"{segmentation_method} is an invalid segmentation method.")

    @classmethod
    def from_trial_directory(cls, trial_dir: str) -> VocalicsComponent:
        vocalics_component_path = f"{trial_dir}/vocalics_component_segmented_utterances.pkl"
        feature_names_path = f"{trial_dir}/features.txt"

        if not os.path.exists(vocalics_component_path):
            raise Exception(f"Could not find the file vocalics_component_segmented_utterances.pkl in {trial_dir}.")

        if not os.path.exists(feature_names_path):
            raise Exception(f"Could not find the file features.txt in {trial_dir}.")

        with open(vocalics_component_path, "rb") as f:
            segmented_utterances = pickle.load(f)

        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)

        return cls(segmented_utterances, feature_names)

    def sparse_series(self, num_time_steps: int, mission_start: datetime) -> VocalicsSparseSeries:
        values = np.zeros((self.segmented_utterances[0].vocalic_series.num_series, num_time_steps))
        mask = np.zeros(num_time_steps)  # 1 for time steps with observation, 0 otherwise
        timestamps: List[Optional[datetime]] = [None] * num_time_steps
        sparse_utterances: List[Optional[SegmentedUtterance]] = [None] * num_time_steps
        previous_from_self: List[Optional[int]] = [None] * num_time_steps
        previous_from_other: List[Optional[int]] = [None] * num_time_steps

        previous_from_subject: Dict[str, int] = {}
        for i, utterance in enumerate(self.segmented_utterances):
            # We consider that the observation is available at the end of an utterance. We take the average vocalics
            # per feature within the utterance as a measurement at the respective time step.
            time_step = int((utterance.end - mission_start).total_seconds())

            if time_step >= num_time_steps:
                msg = f"Time step {time_step} exceeds the number of time steps {num_time_steps} at " \
                      f"utterance {i} out of {len(self.segmented_utterances)} ending at " \
                      f"{utterance.end.isoformat()} considering an initial timestamp of {mission_start.isoformat()}."
                logger.warning(msg)
                break

            if mask[time_step] == 1:
                # A previous utterance finished at the same time. Discard this one.
                continue

            values[:, time_step] = utterance.vocalic_series.values.mean(axis=1)
            mask[time_step] = 1
            timestamps[time_step] = utterance.end
            sparse_utterances[time_step] = utterance
            previous_from_self[time_step] = previous_from_subject.get(utterance.subject_id, None)

            # Find most recent time another subject spoke if any
            most_recent_time = 0
            for subject, time in previous_from_subject.items():
                if subject == utterance.subject_id:
                    continue

                if time > most_recent_time:
                    most_recent_time = time
                    previous_from_other[time_step] = time

            previous_from_subject[utterance.subject_id] = time_step

        return VocalicsSparseSeries(values=values, mask=mask, timestamps=timestamps, utterances=sparse_utterances,
                                    previous_from_self=previous_from_self, previous_from_other=previous_from_other)

    def plot_features(self, axs: List[Any], num_time_steps: int, mission_start: datetime,
                      timestamp_as_index: bool = True, normalize: bool = False):
        sparse_series = self.sparse_series(num_time_steps, mission_start)
        if normalize:
            sparse_series.normalize_per_subject()
        plot_vocalic_features(axs, sparse_series, self.features, timestamp_as_index)

    def plot_utterance_durations(self, ax: Any):
        utterances_per_subject: Dict[str, List[SegmentedUtterance]] = {}
        for utterance in self.segmented_utterances:
            if utterance.subject_id not in utterances_per_subject:
                utterances_per_subject[utterance.subject_id] = [utterance]
            else:
                utterances_per_subject[utterance.subject_id].append(utterance)

        return plot_utterance_durations(ax, utterances_per_subject)

    @classmethod
    def _split_with_current_utterance_truncation(cls, vocalics: Vocalics, min_vocalic_values: int) -> VocalicsComponent:
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

        segmented_utterances: List[SegmentedUtterance] = []
        for i, current_utterance in enumerate(utterances):
            # Start a new segmented utterance if the previous one is completed or not available
            segmented_utterance = SegmentedUtterance.from_utterance(current_utterance, current_utterance.start,
                                                                    current_utterance.end)

            next_utterance = utterances[i + 1] if i < len(utterances) - 1 else None

            if next_utterance is None or next_utterance.start >= current_utterance.end:
                # No overlap. Use the full vocalics
                segmented_utterance.vocalic_series = copy.deepcopy(current_utterance.vocalic_series)
            else:
                # Add vocalics until it overlaps with the start of the next utterance.
                j = 0
                while j < current_utterance.vocalic_series.size and next_utterance.start >= \
                        current_utterance.vocalic_series.timestamps[j]:
                    # Stop when vocalics start to overlap with the start of the next utterance
                    j += 1

                segmented_utterance.vocalic_series = current_utterance.vocalic_series.get_time_range(0, j)
                segmented_utterance.end = next_utterance.start

            if segmented_utterance.vocalic_series.size >= min_vocalic_values:
                segmented_utterances.append(segmented_utterance)
            else:
                msg = f"Segmented utterance at {segmented_utterance.start.isoformat()} and ending " \
                      f"at {segmented_utterance.end.isoformat()} was not added to a series because it has " \
                      f"{segmented_utterance.vocalic_series.size} vocalic values which is less " \
                      f"than the minimum required of {min_vocalic_values}. Text: {segmented_utterance.text}"
                logger.warning(msg)

        return cls(segmented_utterances, vocalics.features)

    @classmethod
    def _split_with_next_utterance_truncation(cls, vocalics: Vocalics, min_vocalic_values: int) -> VocalicsComponent:
        raise RuntimeError("Not implemented")

    @classmethod
    def _split_with_overlap(cls, vocalics: Vocalics, min_vocalic_values: int) -> VocalicsComponent:
        """
        Split utterances from different individuals into two series. Merging utterances of the current speaker in the
        active series, and changing series when another person is the speaker. We keep all utterances including any
        overlapping pieces.
        """

        # Single list with the utterances from all the subjects
        utterances: List[Utterance] = []
        for u in vocalics.utterances_per_subject.values():
            utterances.extend(u)

        # Remove utterances with no vocalics
        utterances = VocalicsComponent._filter_utterances(utterances)

        # We sort the utterances by when participants stop speaking, regardless of the subject they belong to.
        utterances.sort(key=lambda utterance: utterance.end)

        segmented_utterances: List[SegmentedUtterance] = []
        for i, current_utterance in enumerate(utterances):
            segmented_utterance = SegmentedUtterance.from_utterance(current_utterance, current_utterance.start,
                                                                    current_utterance.end)
            segmented_utterance.vocalic_series = copy.deepcopy(current_utterance.vocalic_series)

            if segmented_utterance.vocalic_series.size >= min_vocalic_values:
                segmented_utterances.append(segmented_utterance)
            else:
                msg = f"Segmented utterance at {segmented_utterance.start.isoformat()} and ending " \
                      f"at {segmented_utterance.end.isoformat()} was not added to a series because it has " \
                      f"{segmented_utterance.vocalic_series.size} vocalic values which is less " \
                      f"than the minimum required of {min_vocalic_values}. Text: {segmented_utterance.text}"
                logger.warning(msg)

        return cls(segmented_utterances, vocalics.features)

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
        with open(f"{out_dir}/vocalics_component_segmented_utterances.pkl", "wb") as f:
            pickle.dump(self.segmented_utterances, f)

        if save_feature_names:
            with open(f"{out_dir}/features.txt", "w") as f:
                json.dump(self.features, f)
