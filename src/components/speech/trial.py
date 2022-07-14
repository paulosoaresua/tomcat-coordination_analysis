import json
from typing import Any, Dict, List

from dateutil.parser import parse
from src.components.speech.common import Utterance
from src.components.speech.vocalics_reader import VocalicsReader
from tqdm import tqdm


class Trial:
    """
    This class is used to parse asr/final transcriptions to extract the initial and final timestamps of each subject's
    utterances. Based on those intervals, we will be able to extract appropriate vocalic features for the specific
    utterances later.
    """

    TRIAL_TOPIC = "trial"
    ASR_TOPIC = "agent/asr/final"
    MISSION_STATE_TOPIC = "observations/events/mission"
    VOCALIC_FEATURES = ["pitch", "intensity"]

    def __init__(self,
                 filepath: str,
                 ignore_outside_mission: bool = True,
                 no_vocalics: bool = False):
        self.utterances_per_subject: Dict[str, List[Utterance]] = {}
        self.subject_ids = []
        self.subject_id_to_color = {}
        self.id = ''
        self.timestamp_offset = None
        self.trial_start = None
        self.trial_end = None
        self.mission_start = None
        self.mission_end = None
        self.ignore_outside_mission = ignore_outside_mission
        self.no_vocalics = no_vocalics

        self._parse_metadata_file(filepath)

    def _parse_metadata_file(self, filepath: str) -> None:
        asr_messages = self._get_asr_msgs_and_store_info(filepath)

        # Verify that these fields are filled
        assert self.id

        if self.ignore_outside_mission:
            assert self.mission_start is not None
            assert self.mission_end is not None

        pbar = tqdm(total=len(asr_messages))
        for asr_message in asr_messages:
            msg_end_timestamp = parse(asr_message["data"]["end_timestamp"])
            msg_start_timestamp = parse(asr_message["data"]["start_timestamp"])

            if self.ignore_outside_mission:
                if msg_end_timestamp < self.mission_start:
                    # Ignore utterances before the trial starts.
                    # If an utterance started before but finished after the trial started,
                    # we include the full utterance in the list anyway
                    pbar.update()
                    continue

                if msg_start_timestamp > self.mission_end:
                    # Stop looking for utterances after the trial ends
                    pbar.n = len(asr_messages)
                    pbar.close()
                    break

            pbar.update()

            # get subject callsign
            subject_callsign = self.subject_id_to_color[asr_message["data"]
                                                        ["participant_id"]]

            text = asr_message["data"]["text"]

            utterance = Utterance(subject_callsign,
                                  msg_start_timestamp,
                                  msg_end_timestamp,
                                  text)

            if subject_callsign in self.utterances_per_subject:
                self.utterances_per_subject[subject_callsign].append(utterance)
            else:
                self.utterances_per_subject[subject_callsign] = [utterance]

        if not self.no_vocalics:
            self._read_vocalic_features_for_utterances()

    def _get_asr_msgs_and_store_info(self, filepath: str) -> List[Dict[str, Any]]:
        asr_messages = []

        with open(filepath, 'r') as f:
            for line in f:
                try:
                    json_message = json.loads(line)

                    topic = json_message.get("topic", "")
                    if topic == Trial.TRIAL_TOPIC:
                        self._store_trial_info(json_message)
                    if topic == Trial.ASR_TOPIC:
                        asr_messages.append(json_message)
                    elif topic == Trial.MISSION_STATE_TOPIC:
                        self._store_mission_state(json_message)
                except:
                    print(f"[ERROR] Bad json line of len: {len(line)}, {line}")

        sorted_asr_messages = sorted(
            asr_messages, key=lambda x: parse(x["header"]["timestamp"])
        )

        return sorted_asr_messages

    def _store_trial_info(self, json_message: Any) -> None:
        if json_message["msg"]["sub_type"] == "start":
            self.trial_start = parse(json_message["msg"]["timestamp"])
            self.id = json_message["msg"]["trial_id"]
            self.number = json_message["data"]["trial_number"]

            # Stores the list of subject ids in the game and the map between ID and color
            self.subject_ids = [subjectId.strip()
                                for subjectId in json_message["data"]["subjects"]]
            for info in json_message["data"]["client_info"]:
                subject_color = info["callsign"].lower()
                subject_id = info["participant_id"]
                self.subject_id_to_color[subject_id] = subject_color

                # The ASR agent might use the subject_name as id sometimes.
                subject_name = info["playername"]
                self.subject_id_to_color[subject_name] = subject_color
        else:
            self._trial_end = parse(json_message["msg"]["timestamp"])

    def _store_mission_state(self, json_message: Any) -> None:
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            self.mission_start = parse(json_message["header"]["timestamp"])
        else:
            self.mission_end = parse(json_message["header"]["timestamp"])

    def _read_vocalic_features_for_utterances(self) -> None:
        reader = VocalicsReader()
        vocalics_per_subject = reader.read(self.id, Trial.VOCALIC_FEATURES)

        earliest_vocalics_timestamp = None
        vocalics_subject_callsign_to_id = {}
        for subject_id, vocalics in vocalics_per_subject.items():
            vocalics_subject_callsign_to_id[self.subject_id_to_color[subject_id]] = subject_id

            if earliest_vocalics_timestamp is None or earliest_vocalics_timestamp > vocalics[0].timestamp:
                earliest_vocalics_timestamp = vocalics[0].timestamp

        self.timestamp_offset = self.trial_start - earliest_vocalics_timestamp

        for subject_callsign in self.utterances_per_subject.keys():
            if subject_callsign not in vocalics_subject_callsign_to_id.keys():
                print(
                    f"[WARN] No vocalic feature found for subject {subject_callsign}.")
                continue

            vocalics = vocalics_per_subject[vocalics_subject_callsign_to_id[subject_callsign]]

            for utterance in self.utterances_per_subject[subject_callsign]:
                num_measurements = 0

                # reset the vocalics index here just in case there are overlapping utterances for a subject
                v = 0

                sum_vocalic_features: Dict[str, float] = {}

                # Find start index of vocalic features that matches the start of an utterance
                while v < len(vocalics) and vocalics[v].timestamp + self.timestamp_offset < utterance.start:
                    v += 1

                # Collect vocalic features within an utterance
                while v < len(vocalics) and vocalics[v].timestamp + self.timestamp_offset <= utterance.end:
                    utterance.vocalic_series.append(vocalics[v])

                    # Accumulate vocalic features to average later
                    for feature_name, feature_value in vocalics[v].features.items():
                        if feature_name not in sum_vocalic_features:
                            sum_vocalic_features[feature_name] = 0.0

                        sum_vocalic_features[feature_name] += feature_value

                    num_measurements += 1
                    v += 1

                # Compute average vocalics
                if num_measurements > 0:
                    for name, value in sum_vocalic_features.items():
                        utterance.average_vocalics[name] = value / \
                            float(num_measurements)

                if num_measurements == 0:
                    subject_callsign = self.subject_id_to_color[subject_id]
                    print(
                        "[WARN] No vocalic features detected for utterance between " +
                        f"{utterance.start.isoformat()} and {utterance.end.isoformat()}" +
                        f" for subject {subject_callsign} in trial {self.id}. Text: {utterance.text}")
