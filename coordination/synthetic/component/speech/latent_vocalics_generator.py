from typing import Dict, List, Optional, Tuple

from datetime import datetime

import numpy as np
import random

from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries


class LatentVocalicsGenerator:
    """
    This class generates synthetic evidence for the vocalics component of a coordination model considering the existence
    of latent and observed vocalic features.
    """

    def __init__(self, coordination_series: np.ndarray, num_vocalic_features: int, time_scale_density: float,
                 num_speakers: int):
        self._coordination_series = coordination_series
        self._num_vocalic_features = num_vocalic_features
        self._time_scale_density = time_scale_density
        self._num_speakers = num_speakers

    def generate(self, seed: Optional[int] = None) -> Tuple[VocalicsSparseSeries, VocalicsSparseSeries]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # mask_a, mask_b = self._generate_random_masks()

        num_time_steps = len(self._coordination_series)

        speakers = self._generate_random_speakers()

        mask = np.zeros(num_time_steps)
        mask[0] = int(speakers[0] is not None)

        # Subjects A and B
        previous_self = [None] * num_time_steps
        previous_other = [None] * num_time_steps
        previous_time_per_speaker: Dict[str, int] = {}
        latent_values = np.zeros((self._num_vocalic_features, num_time_steps))
        observed_values = np.zeros((self._num_vocalic_features, num_time_steps))
        utterances: List[Optional[SegmentedUtterance]] = [None] * num_time_steps
        for t in range(num_time_steps):
            current_coordination = self._coordination_series[t]

            if speakers[t] is not None:
                mask[t] = 1

                previous_time_self = previous_time_per_speaker.get(speakers[t], None)
                previous_time_other = None
                for speaker, time in previous_time_per_speaker.items():
                    if speaker == speakers[t]:
                        continue

                    # Most recent vocalics from a different speaker
                    previous_time_other = time if previous_time_other is None else max(previous_time_other, time)

                previous_value_self = None if previous_time_self is None else latent_values[:, previous_time_self]
                previous_value_other = None if previous_time_other is None else latent_values[:, previous_time_other]

                latent_values[:, t] = self._sample_latent(previous_value_self, previous_value_other,
                                                          current_coordination)
                observed_values[:, t] = self._sample_observed(latent_values[:, t])

                previous_self[t] = previous_time_self
                previous_other[t] = previous_time_other

                # Dummy utterance
                utterances[t] = SegmentedUtterance(f"Speaker {speakers[t]}", datetime.now(), datetime.now(), "")
                previous_time_per_speaker[speakers[t]] = t

        latent_series = VocalicsSparseSeries(utterances=utterances, previous_from_self=previous_self,
                                             previous_from_other=previous_other, values=latent_values, mask=mask)
        observed_series = VocalicsSparseSeries(utterances=utterances, previous_from_self=previous_self,
                                               previous_from_other=previous_other, values=observed_values, mask=mask)

        return latent_series, observed_series

    def _generate_random_speakers(self) -> List[Optional[str]]:
        # We always change speakers between time steps when generating vocalics
        transition_matrix = 1 - np.eye(self._num_speakers + 1)

        transition_matrix *= self._time_scale_density / (self._num_speakers - 1)
        transition_matrix[:-1, -1] = 1 - self._time_scale_density

        prior = np.ones(self._num_speakers + 1) * self._time_scale_density / self._num_speakers
        prior[-1] = 1 - self._time_scale_density

        initial_speaker = np.random.choice(self._num_speakers + 1, 1, p=prior)[0]
        initial_speaker = None if initial_speaker == self._num_speakers else initial_speaker
        speakers = [initial_speaker]

        num_time_steps = len(self._coordination_series)
        for t in range(1, num_time_steps):
            probabilities = transition_matrix[self._num_speakers] if speakers[t - 1] is None else transition_matrix[
                speakers[t - 1]]
            speaker = np.random.choice(self._num_speakers + 1, 1, p=probabilities)[0]
            speaker = None if speaker == self._num_speakers else speaker
            speakers.append(speaker)

        return speakers

    def _sample_latent(self, previous_self: Optional[float], previous_other: Optional[float],
                       coordination: float) -> np.ndarray:
        raise NotImplementedError

    def _sample_observed(self, latent_vocalics: np.array) -> np.ndarray:
        raise NotImplementedError
