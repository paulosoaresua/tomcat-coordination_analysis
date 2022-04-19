from typing import List


class Vocalics:

    def __init__(self, timestamp: Any, features: Dict[str, float]):
        self.timestamp = timestamp
        self.features = features


class Utterance:

    def __init__(self, start: Any, end: Any):
        # Initial and final timestamp of an utterance
        self.start = start
        self.end = end
        self.vocalic_series: List[Vocalics] = []
