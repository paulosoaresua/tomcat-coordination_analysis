from enum import Enum
from typing import Dict, List

import numpy as np
from src.components.speech.common import (SegmentedUtterance, Utterance,
                                          VocalicsComponent)


class VocalicsAggregator:
    class SplitMethod(Enum):
        # If the current utterance overlaps with the next, it ends the current utterance when the next one starts
        TRUNCATE_CURRENT = 1

        # If the current utterance overlaps with the next, it starts the next utterance when the current one finishes.
        # The next utterance will be discarded if the current utterance finishes after the next one.
        TRUNCATE_NEXT = 2

        # It considers the entire current and next utterances regardless of whether they overlap.
        OVERLAP = 3

    def __init__(self, utterances_per_subject: Dict[str, List[Utterance]]) -> None:
        """Aggregate vocalic features from speech

        Args:
            utterances_per_subject (Dict[str, List[Utterance]]): utterances of each subject
        """
        self._utterances_per_subject = utterances_per_subject

    def split(self, method: SplitMethod) -> VocalicsComponent:
        """Dispatch splitting vocalic feature method

        Args:
            method (SplitMethod): method for splitting vocalic feature

        Raises:
            RuntimeError: unknown split method

        Returns:
            VocalicsComponent: vocalics component
        """
        if method == VocalicsAggregator.SplitMethod.TRUNCATE_CURRENT:
            return self._split_with_current_utterance_truncation()
        elif method == VocalicsAggregator.SplitMethod.TRUNCATE_NEXT:
            return self._split_with_next_utterance_truncation()
        elif method == VocalicsAggregator.SplitMethod.OVERLAP:
            return self._split_with_overlap()
        else:
            raise RuntimeError("invalid split method choice")

    def _split_with_current_utterance_truncation(self) -> VocalicsComponent:
        """Split vocalic feature and truncate overlapped utterances.

        Returns:
            VocalicsComponent: vocalics component
        """
        utterances = []
        for u in self._utterances_per_subject.values():
            utterances.extend(u)

        # remove any utterance with no vocalic features
        utterances = list(filter(lambda u: len(u.vocalic_series) > 0, utterances))

        # sort utterances by timestamp, regardless of the subject it belongs to.
        utterances.sort(key=lambda utterance: utterance.start)

        series_a: List[SegmentedUtterance] = []
        series_b: List[SegmentedUtterance] = []

        turn_a = True
        raw_vocalics: Dict[str, List[float]] = {}
        segmented_utterance = None
        for i, current_utterance in enumerate(utterances):
            # start a new segmented utterance if the previous one is completed or not available
            if segmented_utterance is None:
                segmented_utterance = SegmentedUtterance(
                    current_utterance.start, current_utterance.end)

            next_utterance = utterances[i+1] if i < len(utterances) - 1 else None

            for vocalics in current_utterance.vocalic_series:
                # if vocalics of current utterance overlaps the next, then move on to next utterance
                if next_utterance is not None and vocalics.timestamp > next_utterance.start:
                    segmented_utterance.end = next_utterance.start
                    break

                # add vocalics values to raw_vocalics
                if len(raw_vocalics) == 0:
                    raw_vocalics = {feature_name: []
                                    for feature_name in vocalics.features.keys()}

                for feature_name, value in vocalics.features.items():
                    raw_vocalics[feature_name].append(value)

            # If the subject of the next utterance is the same, we merge the segments by keep updating
            # raw_vocalics and segmented_utterance in the next loop cycle until we encounter utterance 
            # of a different subject
            if next_utterance is None or next_utterance.subject_callsign != current_utterance.subject_callsign:
                for feature_name, values in raw_vocalics.items():
                    # Average the vocalics within an utterance
                    segmented_utterance.average_vocalics[feature_name] = np.mean(
                        values)

                if turn_a:
                    series_a.append(segmented_utterance)
                else:
                    series_b.append(segmented_utterance)

                segmented_utterance = None
                raw_vocalics = {}
                turn_a = not turn_a

        return VocalicsComponent(series_a, series_b)

    def _split_with_next_utterance_truncation(self):
        raise RuntimeError("Not implemented")

    def _split_with_overlap(self):
        raise RuntimeError("Not implemented")
