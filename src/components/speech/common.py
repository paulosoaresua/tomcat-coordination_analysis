from typing import Any, Dict, List


class Vocalics:

    def __init__(self, timestamp: Any, features: Dict[str, float]):
        self.timestamp = timestamp
        self.features = features


class Utterance:

    def __init__(self, subject_id: str, start: Any, end: Any):
        self.subject_id = subject_id
        self.start = start
        self.end = end
        self.vocalic_series: List[Vocalics] = []
        # A value per feature
        self.average_vocalics: Dict[str, float] = {}


class SegmentedUtterance:

    def __init__(self, start: Any, end: Any):
        self.start = start
        self.end = end
        # A value per feature
        self.average_vocalics: Dict[str, float] = {}


class VocalicsComponent:

    def __init__(self, series_a: List[SegmentedUtterance], series_b: List[SegmentedUtterance]):
        # Equalizes the sizes of the two series
        # TODO - fix this. The series do not need to have the same size. Changing this affects the VocalicsWriter's
        #  logic.
        self.size = min(len(series_a), len(series_b))
        self.series_a = series_a[:self.size]
        self.series_b = series_b[:self.size]

        if self.size > 0:
            self.feature_names = [feature_name for feature_name in self.series_a[0].average_vocalics.keys()]
        else:
            self.feature_names = []

