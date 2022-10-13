from typing import List

from coordination.component.speech.vocalics_component import VocalicsSparseSeries


class SeriesData:

    def __init__(self, vocalics: VocalicsSparseSeries):
        self.vocalics = vocalics

    @property
    def num_time_steps(self):
        return self.vocalics.num_time_steps

class Dataset:

    def __init__(self, series: List[SeriesData]):
        self.series = series

    @property
    def num_trials(self):
        return len(self.series)
