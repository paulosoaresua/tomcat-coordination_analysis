from typing import Tuple, Any, List, Dict
from enum import Enum
import numpy as np
from src.components.speech.common import Utterance, SegmentedUtterance, VocalicsComponent


class VocalicsAggregator:
    class SplitMethod(Enum):
        # If the current utterance overlaps with the next, it ends the current utterance when the next one starts
        TRUNCATE_CURRENT = 1

        # If the current utterance overlaps with the next, it starts the next utterance when the current one finishes.
        # The next utterance will be discarded if the current utterance finishes after the next one.
        TRUNCATE_NEXT = 2

        # It considers the entire current and next utterances regardless of whether they overlap.
        OVERLAP = 3

    def __init__(self, utterances_per_subject: Dict[str, List[Utterance]]):
        self._utterances_per_subject = utterances_per_subject

    def split(self, method: SplitMethod):
        if method == VocalicsAggregator.SplitMethod.TRUNCATE_CURRENT:
            self.split_with_current_utterance_truncation()
        elif method == VocalicsAggregator.SplitMethod.TRUNCATE_CURRENT:
            self.split_with_next_utterance_truncation()
        else:
            self.split_with_overlap()

    def split_with_current_utterance_truncation(self):
        utterances = []
        for _, u in self._utterances_per_subject.items():
            utterances.extend(u)

        # First, we sort utterances by timestamp, regardless of the subject it belongs to.
        utterances.sort(key=lambda utterance: utterance.timestamp)

        series_a: List[SegmentedUtterance] = []
        series_b: List[SegmentedUtterance] = []
        turn_a = True
        raw_vocalics: Dict[str, List[float]] = {}
        segmented_utterance = None
        for i, current_utterance in enumerate(utterances):
            if segmented_utterance is None:
                segmented_utterance = SegmentedUtterance(current_utterance.start, current_utterance.end)
                series = {}
            next_utterance = utterances[i+1] if i < len(utterances) - 1 else None
            for j, vocalics in enumerate(current_utterance.vocalic_series):
                if next_utterance is not None and vocalics.timestamp > next_utterance.start:
                    segmented_utterance.end = next_utterance.start
                    break
                else:
                    if len(raw_vocalics) == 0:
                        raw_vocalics = {feature_name: [] for feature_name in vocalics.features.keys()}

                    for feature_name, value in vocalics.features.items():
                        raw_vocalics[feature_name].append(value)

                if next_utterance is None or next_utterance.subject_id != current_utterance.subject_id:
                    # If the subject of the next utterance is the same, we merge the segments
                    for feature_name, values in raw_vocalics.items():
                        # Average the vocalics within an utterance
                        segmented_utterance.average_vocalics[feature_name] = np.mean(values)

                    if turn_a:
                        series_a.append(segmented_utterance)
                    else:
                        series_b.append(segmented_utterance)

                    segmented_utterance = None
                    turn_a = not turn_a

        return VocalicsComponent(series_a, series_b)

    def split_with_next_utterance_truncation(self):
        pass

    def split_with_overlap(self):
        pass

