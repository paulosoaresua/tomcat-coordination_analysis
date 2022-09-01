from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from glob import glob
from datetime import datetime
import logging
import math
import os
import pickle
import pydub
import re

import numpy as np
from scipy.io import wavfile

from coordination.entity.trial_metadata import TrialMetadata
from coordination.component.speech.vocalics_component import SegmentedUtterance, VocalicsComponent

logger = logging.getLogger()


class AudioSegment:

    def __init__(self, source: str, start: datetime, end: datetime, data: np.ndarray, sample_rate: int):
        self.source = source
        self.start = start
        self.end = end
        self.data = data
        self.sample_rate = sample_rate

    def save_to_wav(self, out_filepath: str):
        wavfile.write(out_filepath, self.sample_rate, self.data)

    def save_to_mp3(self, out_filepath: str, normalized: bool = False):
        # Code from  https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file- ...
        # into-a-numpy-array-save-a-numpy-array-to-mp3
        channels = 2 if (self.data.ndim == 2 and self.data.shape[1] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(self.data * 2 ** 15)
        else:
            y = np.int16(self.data)
        mp3_audio = pydub.AudioSegment(y.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=channels)
        mp3_audio.export(out_filepath, format="mp3", bitrate="320k")


class AudioSparseSeries:
    def __init__(self, audio_segments: List[Optional[AudioSegment]]):
        self.audio_segments = audio_segments


class VocalicsComponentAudio:

    def __init__(self, series_a: List[AudioSegment], series_b: List[AudioSegment]):
        self.series_a = series_a
        self.series_b = series_b

    @classmethod
    def from_vocalics_component(cls, trial_audio: TrialAudio,
                                vocalics_component: VocalicsComponent) -> VocalicsComponentAudio:
        def segment_audio_from_segmented_utterances(utterances: List[SegmentedUtterance]) -> List[AudioSegment]:
            audio_segments: List[AudioSegment] = []
            for utterance in utterances:
                audio_series = trial_audio.audio_per_participant[utterance.subject_id]
                audio_data_segment = audio_series.get_data_segment(utterance.start, utterance.end)
                audio_segment = AudioSegment(utterance.subject_id, utterance.start, utterance.end,
                                             audio_data_segment, audio_series.sample_rate)
                audio_segments.append(audio_segment)

            return audio_segments

        audio_series_a = segment_audio_from_segmented_utterances(vocalics_component.series_a)
        audio_series_b = segment_audio_from_segmented_utterances(vocalics_component.series_b)
        return VocalicsComponentAudio(audio_series_a, audio_series_b)

    @classmethod
    def from_trial_directory(cls, trial_dir: str) -> VocalicsComponentAudio:
        vocalics_component_a_path = f"{trial_dir}/vocalics_component_audio_a.pkl"
        vocalics_component_b_path = f"{trial_dir}/vocalics_component_audio_b.pkl"

        if not os.path.exists(vocalics_component_a_path):
            raise Exception(f"Could not find the file vocalics_component_audio_a.pkl in {trial_dir}.")

        if not os.path.exists(vocalics_component_b_path):
            raise Exception(f"Could not find the file vocalics_component_audio_b.pkl in {trial_dir}.")

        with open(vocalics_component_a_path, "rb") as f:
            series_a = pickle.load(f)

        with open(vocalics_component_b_path, "rb") as f:
            series_b = pickle.load(f)

        return cls(series_a, series_b)

    def sparse_series(self, num_time_steps: int) -> Tuple[AudioSparseSeries, AudioSparseSeries]:
        def series_to_seconds(audio_segments: List[AudioSegment], initial_timestamp: datetime) -> AudioSparseSeries:
            segments: List[Optional[AudioSegment]] = [None] * num_time_steps

            for i, audio_segment in enumerate(audio_segments):
                # We consider that the observation is available at the end of an utterance. We take the average vocalics
                # per feature within the utterance as a measurement at the respective time step.
                time_step = int((audio_segment.end - initial_timestamp).total_seconds())
                if time_step >= num_time_steps:
                    logger.warning(f"""Time step {time_step} exceeds the number of time steps {num_time_steps} at 
                                   audio segment {i} out of {len(audio_segments)} ending at 
                                   {audio_segment.end.isoformat()} considering an initial timestamp 
                                   of {initial_timestamp.isoformat()}.""")
                    break

                segments[time_step] = audio_segment

            return AudioSparseSeries(segments)

        # The first audio always goes in series A
        earliest_timestamp = self.series_a[0].start
        sparse_series_a = series_to_seconds(self.series_a, earliest_timestamp)
        sparse_series_b = series_to_seconds(self.series_b, earliest_timestamp)

        return sparse_series_a, sparse_series_b

    def save(self, out_dir: str):
        with open(f"{out_dir}/vocalics_component_audio_a.pkl", "wb") as f:
            pickle.dump(self.series_a, f)

        with open(f"{out_dir}/vocalics_component_audio_b.pkl", "wb") as f:
            pickle.dump(self.series_b, f)


class AudioSeries:

    def __init__(self, sample_rate: int, data: np.ndarray, baseline_timestamp: datetime):
        self.sample_rate = sample_rate
        self.data = data
        self.baseline_timestamp = baseline_timestamp

    def get_data_segment(self, start: datetime, end: datetime):
        start_time_step = (start - self.baseline_timestamp).total_seconds()
        end_time_step = (end - self.baseline_timestamp).total_seconds() + 1

        lower_idx = math.floor(self.sample_rate * start_time_step)
        upper_idx = min(math.ceil(self.sample_rate * end_time_step), len(self.data))
        return self.data[lower_idx: upper_idx]


class TrialAudio:

    def __init__(self, trial_metadata: TrialMetadata, audio_dir: str):
        self._trial_metadata = trial_metadata
        self.audio_per_participant: Dict[str, AudioSeries] = {}
        self._read_audio_files(audio_dir)

    def _read_audio_files(self, audio_dir: str):
        self.audio_per_participant = {}
        filepaths = glob(f"{audio_dir}/*Trial-{self._trial_metadata.number}*.wav")
        for filepath in filepaths:
            result = re.search(r".+_Member-(.+)_CondBtwn.+", filepath)
            subject_id = result.group(1)
            audio_series = AudioSeries(*wavfile.read(filepath), baseline_timestamp=self._trial_metadata.trial_start)
            self.audio_per_participant[self._trial_metadata.subject_id_map[subject_id]] = audio_series

# if __name__ == "__main__":
#     SegmentedAudio.from_segmented_utterances(
#         "/Users/paulosoares/data/study-3_2022/audio/HSRData_ClientAudio_Trial-T000745_Team-TM000273_Member-E000771_CondBtwn-ASI-UAZ-TA1_CondWin-na_Vers-1.wav",
#         [])
