from __future__ import annotations
from typing import List, Optional

from datetime import datetime

import numpy as np

from coordination.common.sparse_series import SparseSeries
from coordination.entity.vocalics import Utterance
from coordination.entity.vocalics_series import VocalicsSeries


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

    def __repr__(self):
        return f"{self.subject_id}: {self.text}"


class VocalicsSparseSeries(SparseSeries):

    def __init__(self, utterances: List[Optional[SegmentedUtterance]],
                 previous_from_self: List[Optional[int]],
                 previous_from_other: List[Optional[int]], *args, **kwargs):
        """
        @param utterances: List of utterances from different subjects
        @param previous_from_self: Most recent timestamp with utterance from the same subject
        @param previous_from_others: Most recent timestamp with utterance from a different subject
        """
        super().__init__(*args, **kwargs)

        self.subjects = set([u.subject_id for u in utterances if u is not None])
        self.utterances = utterances
        self.previous_from_self = previous_from_self
        self.previous_from_other = previous_from_other

    def normalize_per_subject(self):
        """
        Make values of series have mean 0 and standard deviation 1 per subject
        """
        self.values = self.values.astype(float)

        for subject in self.subjects:
            valid_indices = [t for t, mask in enumerate(self.mask) if
                             mask == 1 and self.utterances[t].subject_id == subject]

            mean = self.values[:, valid_indices].mean(axis=1)[:, np.newaxis]
            std = self.values[:, valid_indices].std(axis=1)[:, np.newaxis]
            self.values[:, valid_indices] = (self.values[:, valid_indices] - mean) / std
