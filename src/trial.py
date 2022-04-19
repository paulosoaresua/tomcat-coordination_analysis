from typing import Dict, Any, List
from dateutil.parser import parse
import json
from src.components.speech import Vocalics, Utterance
from src.vocalics_reader import VocalicsReader


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

    def __init__(self):
        self.utterances_per_subject: Dict[str, List[Utterance]] = {}
        self.subject_ids = []
        self.subject_id_to_color = []
        self.id = ""
        self.start = None
        self.end = None

    def parse(self, filepath: str):
        asr_messages = self._get_sorted_asr_messages(filepath)

        for asr_message in asr_messages:
            timestamp = parse(asr_message["header"]["timestamp"])
            if timestamp < self.start:
                # Ignore utterances before the trial starts
                continue
            if timestamp > self.end:
                # Stop looking for utterances after the trial ends
                break

            # If an utterance started before but finished after the trial started, we include the full utterance in the
            # list anyway
            utterance = Utterance(parse(asr_message["data"]["start"]), parse(asr_message["data"]["end"]))

            subject_id = asr_message["data"]["subject_id"]
            if subject_id in self.utterances_per_subject:
                self.utterances_per_subject[subject_id].append(utterance)
            else:
                self.utterances_per_subject[subject_id] = [utterance]

        self._read_vocalic_features()

    def _get_sorted_asr_messages(self, filepath: str):
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
                    print(f"Bad json line of len: {len(line)}, {line}")

        sorted_messages = sorted(
            asr_messages, key=lambda x: parse(x["header"]["timestamp"])
        )

        return sorted_messages

    def _store_trial_info(self, json_message: Any):
        self.id = json_message["msd"]["trial_id"]
        self.number = json_message["data"]["trial_number"]

        # Stores the list of subject ids in the game and the map between ID and color
        self.subject_ids = [subjectId.strip() for subjectId in json_message["data"]["subjects"]]
        for info in json_message["data"]["client_info"]:
            subject_color = info["callsign"].lower()
            subject_id = info["subject_id"]
            self.subject_id_to_color[subject_id] = subject_color

            # The ASR agent might use the subject_name as id sometimes.
            subject_name = info["subjectname"]
            self.subject_id_to_color[subject_name] = subject_color

    def _store_mission_state(self, json_message: Any):
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            self.start = parse(json_message["header"]["timestamp"])
        else:
            self.end = parse(json_message["header"]["timestamp"])

    def _read_vocalic_features(self):
        reader = VocalicsReader()
        reader.read(self.id, self.start, self.end, Trial.VOCALIC_FEATURES)

        for subject_id, utterances in self.utterances_per_subject:
            if subject_id not in reader.vocalics_per_subject:
                print(f"No vocalic feature found for subject {subject_id}.")
                break
            vocalics = reader.vocalics_per_subject[subject_id]

            v = 0
            for utterance in utterances:
                num_measurements = 0

                # Find start index of vocalic features that matches the start of an utterance
                while v < len(vocalics) and vocalics[v].timestamp < utterance.start:
                    v += 1

                # Collect vocalic features within an utterance
                while v < len(vocalics) and vocalics[v].timestamp <= utterance.end:
                    utterance.vocalic_series.append(vocalics[v])
                    num_measurements += 1
                    v += 1

                if num_measurements == 0:
                    print(
                        f"No vocalic features detected for utterance between {utterance.start} an {utterance.end} for"
                        " subject {subject_id} in trial {self.id}.")

# if __name__ == '__main__':
#     trial = Trial()
#     trial.parse(
#         "../data/study3/Test_A_Thon/TrialMessages_CondBtwn-1_CondWin-Saturn-StaticMap_Trial-1_Team-55cd3a31_Member-Aptiomiomer1_Vers-3.5.1-dev.84-1c1474c.metadata")
#
#     print(f"{trial.start} - {trial.end}")
#     for subject_id in trial.subject_ids:
#         print(f"{subject_id} has {len(trial.utterances[subject_id])} utterances in the trial")
