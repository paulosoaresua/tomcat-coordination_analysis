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

    def __init__(self, filepath: str):
        self.utterances_per_subject: Dict[str, List[Utterance]] = {}
        self.subject_ids = []
        self.subject_id_to_color = []
        self.id = ''
        self.start = None
        self.end = None

        self._parse_metadata_file(filepath)

    def _parse_metadata_file(self, filepath: str) -> None:
        asr_messages = self._get_asr_msgs_and_store_info(filepath)

        # Verify that these fields are filled
        assert self.id
        assert self.start is not None
        assert self.end is not None

        pbar = tqdm(total = len(asr_messages))
        for asr_message in asr_messages:
            # Comparing header timestamp
            header_timestamp = parse(asr_message["header"]["timestamp"])
            msg_end_timestamp = parse(asr_message["data"]["end_timestamp"])
            if header_timestamp < self.start and msg_end_timestamp > self.start:
                # Ignore utterances before the trial starts.
                # If an utterance started before but finished after the trial started,
                # we include the full utterance in the list anyway
                pbar.update()
                continue
            if header_timestamp > self.end:
                # Stop looking for utterances after the trial ends
                pbar.n = len(asr_messages)
                pbar.close()
                break

            pbar.update()

            subject_id = asr_message["data"]["participant_id"]
            msg_start_timestamp = parse(asr_message["data"]["start_timestamp"])
            text = asr_message["data"]["text"]

            utterance = Utterance(subject_id,
                                  msg_start_timestamp,
                                  msg_end_timestamp,
                                  text)

            if subject_id in self.utterances_per_subject:
                self.utterances_per_subject[subject_id].append(utterance)
            else:
                self.utterances_per_subject[subject_id] = [utterance]

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
        self.id = json_message["msg"]["trial_id"]
        self.number = json_message["data"]["trial_number"]

        # Stores the list of subject ids in the game and the map between ID and color
        self.subject_ids = [subjectId.strip()
                            for subjectId in json_message["data"]["subjects"]]
        for info in json_message["data"]["client_info"]:
            subject_color = info["callsign"].lower()
            subject_id = info["subject_id"]
            self.subject_id_to_color[subject_id] = subject_color

            # The ASR agent might use the subject_name as id sometimes.
            subject_name = info["subjectname"]
            self.subject_id_to_color[subject_name] = subject_color

    def _store_mission_state(self, json_message: Any) -> None:
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            self.start = parse(json_message["header"]["timestamp"])
        else:
            self.end = parse(json_message["header"]["timestamp"])

    def _read_vocalic_features_for_utterances(self) -> None:
        reader = VocalicsReader()
        vocalics_per_subject = reader.read(
            self.id, self.start, self.end, Trial.VOCALIC_FEATURES)

        for subject_id in self.utterances_per_subject.keys():
            if subject_id not in vocalics_per_subject:
                print(
                    f"[WARN] No vocalic feature found for subject {subject_id}.")
                continue

            vocalics = vocalics_per_subject[subject_id]

            for u, utterance in enumerate(self.utterances_per_subject[subject_id]):
                num_measurements = 0

                # reset the vocalics index here just in case there are overlapping utterances for a subject
                v = 0

                sum_vocalic_features: Dict[str, float] = {}

                # Find start index of vocalic features that matches the start of an utterance
                while v < len(vocalics) and vocalics[v].timestamp <= utterance.start:
                    v += 1

                # Collect vocalic features within an utterance
                while v < len(vocalics) and vocalics[v].timestamp <= utterance.end:
                    self.utterances_per_subject[subject_id][u].vocalic_series.append(vocalics[v])

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
                        self.utterances_per_subject[subject_id][u].average_vocalics[name] = value / float(num_measurements)

                if num_measurements == 0:
                    print(
                        "[WARN] No vocalic features detected for utterance between " +
                        f"{utterance.start.isoformat()} and {utterance.end.isoformat()}" + 
                        f" for subject {subject_id} in trial {self.id}.")
