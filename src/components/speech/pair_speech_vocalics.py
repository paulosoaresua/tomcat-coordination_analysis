from typing import List, Dict

import numpy as np
from tqdm import tqdm

from .vocalics_reader import Vocalics
from .metadata_reader import Speech


class Utterance:
    def __init__(self, speech: Speech, vocalics_series: List[Vocalics]):
        self.speech = speech
        self.vocalics_series = vocalics_series

    @property
    def vocalics_matrix(self) -> np.ndarray:
        return np.array([])


def pair_speech_vocalics(speeches_per_subject: Dict[str, List[Speech]],
                         vocalics_per_subject: Dict[str, List[Vocalics]],
                         silent_warnings: bool = False):
    subjects = speeches_per_subject.keys()

    utterances_per_subject = {}
    for subject in subjects:
        if subject not in vocalics_per_subject:
            print(f"[WARN] No vocalic feature found for subject {subject}.")
            continue

        subject_vocalics = vocalics_per_subject[subject]
        subject_speeches = speeches_per_subject[subject]

        t = 0
        pbar = tqdm(total=len(subject_speeches), desc=f"Extract {subject}'s utterances")
        for speech in subject_speeches:
            # Find start index of vocalic features that matches the start of a speech
            while t < len(subject_vocalics) and subject_vocalics[t].timestamp < speech.start_timestamp:
                t += 1

            # Collect vocalic feature values within a speech
            vocalics_in_speech = []
            while t < len(subject_vocalics) and subject_vocalics[t].timestamp <= speech.end_timestamp:
                vocalics_in_speech.append(subject_vocalics[t])
                t += 1

            if len(vocalics_in_speech) > 0:
                utterance = Utterance(speech, vocalics_in_speech)

                if subject not in utterances_per_subject:
                    utterances_per_subject[subject] = []

                utterances_per_subject[subject].append(utterance)
            elif not silent_warnings:
                pbar.write(
                    "[WARN] No vocalic features detected for utterance between " +
                    f"{speech.start_timestamp.isoformat()} and {speech.end_timestamp.isoformat()}" +
                    f" for subject {subject}. Text: {speech.text}")

            pbar.update()
