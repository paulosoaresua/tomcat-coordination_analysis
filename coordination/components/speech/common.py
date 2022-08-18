from datetime import datetime
from typing import Dict, List


class Vocalics:
    def __init__(self, timestamp: datetime, features: Dict[str, float]):
        self.timestamp = timestamp
        self.features = features


class Utterance:
    def __init__(self,
                 subject_callsign: str,
                 start: datetime,
                 end: datetime,
                 text: str):
        self.subject_callsign = subject_callsign
        self.start = start
        self.end = end
        self.text = text

        # Avoid creating default parameters for vocalic_series and average_vocalics.
        # For unknown reasons, the default lists are tied together, changing one also
        # changes the other.
        self.vocalic_series: List[Vocalics] = []
        # A value per feature
        self.average_vocalics: Dict[str, float] = {}


class SegmentedUtterance:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end
        # A value per feature
        self.average_vocalics: Dict[str, float] = {}


class VocalicsComponent:
    def __init__(self, series_a: List[SegmentedUtterance], series_b: List[SegmentedUtterance]):
        self.series_a = series_a
        self.series_b = series_b

        # assuming that series a and series b have the same vocalics types
        if len(self.series_a) > 0:
            self.feature_names = list(self.series_a[0].average_vocalics.keys())
        elif len(self.series_b) > 0:
            self.feature_names = list(self.series_b[0].average_vocalics.keys())
        else:
            self.feature_names = []
