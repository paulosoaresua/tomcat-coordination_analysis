from __future__ import annotations
from typing import Any, Dict, List

from datetime import datetime
from dateutil.parser import parse
import json
import logging
import pickle
import os

from tqdm import tqdm
import numpy as np

from coordination.entity.trial_metadata import TrialMetadata
from coordination.entity.vocalics_series import VocalicsSeries
from coordination.config.database_config import DatabaseConfig
from coordination.loader.vocalics_reader import VocalicsReader

logger = logging.getLogger()


class Utterance:
    def __init__(self, participant_id: str, text: str, start: datetime, end: datetime):
        self.subject_id = participant_id
        self.text = text
        self.start = start
        self.end = end

        # This will contain values for different vocalic features within an utterance.
        # The series is a matrix, with as many rows as the number of features and as many
        # columns as the number of vocalic values in the utterance
        self.vocalic_series: VocalicsSeries = VocalicsSeries(np.array([]), [])

    @property
    def duration_in_seconds(self):
        return (self.end - self.start).total_seconds()


class Vocalics:
    def __init__(self, features: List[str], utterances_per_subject: Dict[str, List[Utterance]]):
        self.features = features
        self.utterances_per_subject = utterances_per_subject

    @classmethod
    def from_asr_messages(cls, asr_messages: List[Any], trial_metadata: TrialMetadata, database_config: DatabaseConfig,
                          features: List[str]) -> Vocalics:
        """
        Parses a list of ASR messages to extract utterances and their corresponding vocalic features.
        """

        asr_messages = sorted(
            asr_messages, key=lambda x: parse(x["header"]["timestamp"])
        )

        utterances_per_subject = {}

        pbar = tqdm(total=len(asr_messages), desc="Parsing utterances")
        for asr_message in asr_messages:
            start_timestamp = parse(asr_message["data"]["start_timestamp"])
            end_timestamp = parse(asr_message["data"]["end_timestamp"])

            if start_timestamp > trial_metadata.mission_end:
                # Stop looking for utterances after the end of a mission
                pbar.n = len(asr_messages)
                pbar.close()
                break

            if end_timestamp < trial_metadata.mission_start:
                # Ignore utterances before the mission starts.
                # If an utterance started before but finished after the trial started,
                # we include the full utterance in the list anyway.
                pbar.update()
                continue

            # Instead of using the subject id, we use it's callsign to be compatible across all components
            subject_id = trial_metadata.subject_id_map[asr_message["data"]["participant_id"]]
            text = asr_message["data"]["text"]

            utterance = Utterance(subject_id,
                                  text,
                                  start_timestamp,
                                  end_timestamp)

            if subject_id in utterances_per_subject:
                utterances_per_subject[subject_id].append(utterance)
            else:
                utterances_per_subject[subject_id] = [utterance]

            pbar.update()

        vocalics_reader = VocalicsReader(database_config, features)
        Vocalics._read_vocalic_features(trial_metadata, utterances_per_subject, vocalics_reader)

        return cls(features, utterances_per_subject)

    @classmethod
    def from_trial_directory(cls, trial_dir: str) -> Vocalics:
        features_path = f"{trial_dir}/features.txt"
        if not os.path.exists(features_path):
            raise Exception(f"Could not find the file features.txt in {features_path}.")

        with open(features_path, "r") as f:
            features = json.load(f)

        vocalics_path = f"{trial_dir}/vocalics.pkl"
        if not os.path.exists(vocalics_path):
            raise Exception(f"Could not find the file vocalics.pkl in {vocalics_path}.")

        with open(vocalics_path, "rb") as f:
            vocalics = pickle.load(f)

        return cls(features, vocalics)

    @staticmethod
    def _read_vocalic_features(trial_metadata: TrialMetadata, utterances_per_subject: Dict[str, List[Utterance]],
                               reader: VocalicsReader):
        """
        Reads vocalic feature values for a series of parsed utterances.
        """

        print("Reading vocalics...")
        # Because we can have utterances that started before the mission and ended after the mission as long as they
        # overlap with times within the mission, we retrieve all the vocalics within a trial instead of passing a
        # time range to the read function.
        vocalics_per_subject = reader.read(trial_metadata, trial_metadata.trial_start)

        if len(vocalics_per_subject) > 0:
            pbar = tqdm(total=len(utterances_per_subject), desc="Adding vocalics to utterances")
            for subject_id in utterances_per_subject.keys():
                if subject_id not in vocalics_per_subject.keys():
                    logger.warning(f"No vocalic features found for subject {subject_id}.")
                    continue

                # Sorted per subject
                vocalic_series = vocalics_per_subject[subject_id]

                t = 0
                for utterance in utterances_per_subject[subject_id]:
                    # Find start index of vocalic features that matches the start of an utterance
                    while t < vocalic_series.size and vocalic_series.timestamps[t] < utterance.start:
                        t += 1

                    # Collect vocalic feature values within an utterance
                    vocalics_in_utterance = []
                    vocalics_timestamps = []
                    while t < vocalic_series.size and vocalic_series.timestamps[t] <= utterance.end:
                        vocalics_timestamps.append(vocalic_series.timestamps[t])
                        vocalics_in_utterance.append(vocalic_series.values[:, t, np.newaxis])
                        t += 1

                    if len(vocalics_in_utterance) > 0:
                        utterance.vocalic_series = VocalicsSeries(np.concatenate(vocalics_in_utterance, axis=1),
                                                                  vocalics_timestamps)
                    else:
                        logger.warning(
                            "No vocalic features detected for utterance between " +
                            f"{utterance.start.isoformat()} and {utterance.end.isoformat()}" +
                            f" for subject {subject_id} in trial {trial_metadata.number}. Text: {utterance.text}")

                pbar.update()

    def save(self, out_dir: str):
        with open(f"{out_dir}/features.txt", "w") as f:
            json.dump(self.features, f)

        with open(f"{out_dir}/vocalics.pkl", "wb") as f:
            pickle.dump(self.utterances_per_subject, f)
