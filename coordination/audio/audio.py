from __future__ import annotations
from typing import Dict, Optional, Union

from glob import glob
from datetime import datetime
import logging
import math
import pydub
import re

import numpy as np
from scipy.io import wavfile

from coordination.component.speech.common import SegmentedUtterance
from coordination.entity.trial_metadata import TrialMetadata
from coordination.entity.vocalics import Utterance


logger = logging.getLogger()


class AudioSegment:

    def __init__(self, source: str, start: datetime, end: datetime, data: np.ndarray, sample_rate: int,
                 transcription: Optional[str] = None):
        self.source = source
        self.start = start
        self.end = end
        self.data = data
        self.sample_rate = sample_rate
        self.transcription = transcription

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


class AudioSeries:

    def __init__(self, source: str, sample_rate: int, data: np.ndarray, baseline_timestamp: datetime):
        self.source = source
        self.sample_rate = sample_rate
        self.data = data
        self.baseline_timestamp = baseline_timestamp

    def get_data_segment(self, start: datetime, end: datetime):
        start_time_step = (start - self.baseline_timestamp).total_seconds()
        end_time_step = (end - self.baseline_timestamp).total_seconds()
        lower_idx = math.floor(self.sample_rate * start_time_step)
        upper_idx = min(math.ceil(self.sample_rate * end_time_step + 1), len(self.data))

        return self.data[lower_idx: upper_idx]

    def get_audio_segment(self, start: datetime, end: datetime, transcription: Optional[str]):
        data = self.get_data_segment(start, end)
        return AudioSegment(self.source, start, end, data, self.sample_rate, transcription)

    def get_audio_segment_from_utterance(self, utterance: Union[SegmentedUtterance, Utterance]):
        return self.get_audio_segment(utterance.start, utterance.end, utterance.text)


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
            frame_rate, data = wavfile.read(filepath)
            audio_series = AudioSeries(subject_id, frame_rate, data,
                                       baseline_timestamp=self._trial_metadata.trial_start)
            self.audio_per_participant[self._trial_metadata.subject_id_map[subject_id].avatar_color] = audio_series
