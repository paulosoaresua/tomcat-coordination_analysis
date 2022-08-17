import json
from datetime import datetime
from typing import Dict, List
from enum import Enum
import pickle
import os

from dateutil.parser import parse
from tqdm import tqdm


class Speech:
    def __init__(self,
                 subject_callsign: str,
                 start_timestamp: datetime,
                 end_timestamp: datetime,
                 text: str) -> None:
        self.subject_callsign = subject_callsign
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.text = text


class MetadataReader:
    # For online inference, we have to change this to match on the header.type and msg.sub_type because the topic
    # won't be available as one of the message's field.
    TRIAL_TOPIC = "trial"
    ASR_TOPIC = "agent/asr/final"
    SCOREBOARD_TOPIC = "observations/events/scoreboard"
    MISSION_STATE_TOPIC = "observations/events/mission"

    class Mode(Enum):
        METADATA = 0
        LOAD = 1

    def __init__(self, mode: Mode, path: str) -> None:
        self.speeches_per_subject: Dict[str, List[Speech]] = {}
        self.subject_id_to_color = {}
        self.id = ''
        self.number = None
        self.trial_start = None
        self.trial_end = None
        self.mission_start = None
        self.mission_end = None
        self.team_score = 0

        if mode == MetadataReader.Mode.METADATA:
            self._parse_metadata_file(path)
        else:
            self._load(path)

    def _parse_metadata_file(self, filepath: str) -> None:
        asr_messages = self._get_asr_msgs_and_store_info(filepath)

        # Verify that these fields are filled
        assert self.subject_ids
        assert self.subject_id_to_color
        assert self.id
        assert self.trial_start
        assert self.trial_end
        assert self.mission_start
        assert self.mission_end

        pbar = tqdm(total=len(asr_messages))
        for asr_message in asr_messages:
            msg_end_timestamp = parse(asr_message["data"]["end_timestamp"])
            msg_start_timestamp = parse(asr_message["data"]["start_timestamp"])

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
            subject_callsign = self.subject_id_to_color[asr_message["data"]["participant_id"]]

            text = asr_message["data"]["text"]

            speech = Speech(subject_callsign,
                            msg_start_timestamp,
                            msg_end_timestamp,
                            text)

            if subject_callsign in self.speeches_per_subject:
                self.speeches_per_subject[subject_callsign].append(speech)
            else:
                self.speeches_per_subject[subject_callsign] = [speech]

    def _get_asr_msgs_and_store_info(self, filepath: str) -> List[Dict[str, Dict]]:
        asr_messages = []

        with open(filepath, 'r') as f:
            for line in f:
                try:
                    json_message = json.loads(line)

                    topic = json_message.get("topic", "")
                    if topic == MetadataReader.TRIAL_TOPIC:
                        self._store_trial_info(json_message)
                    if topic == MetadataReader.ASR_TOPIC:
                        asr_messages.append(json_message)
                    elif topic == MetadataReader.MISSION_STATE_TOPIC:
                        self._store_mission_state(json_message)
                    elif topic == MetadataReader.SCOREBOARD_TOPIC:
                        self._store_team_score(json_message)
                except:
                    print(f"[ERROR] Bad json line of len: {len(line)}, {line}")

        sorted_asr_messages = sorted(
            asr_messages, key=lambda x: parse(x["header"]["timestamp"])
        )

        return sorted_asr_messages

    def _store_trial_info(self, json_message: Dict) -> None:
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
            self.trial_end = parse(json_message["msg"]["timestamp"])

    def _store_mission_state(self, json_message: Dict) -> None:
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            self.mission_start = parse(json_message["header"]["timestamp"])
        else:
            self.mission_end = parse(json_message["header"]["timestamp"])

    def _store_team_score(self, json_message: Dict):
        self.team_score = max(int(json_message["data"]["scoreboard"]["TeamScore"]), self.team_score)

    def _load(self, save_directory: str):
        info_path = f"{save_directory}/info.json"
        if not os.path.exists(info_path):
            raise Exception(f"Could not find the file info.json in {save_directory}.")

        with open(info_path, "r") as f:
            info_dict = json.load(f)
            self.trial_id = info_dict["id"],
            self.number = info_dict["number"]
            self.trial_start = parse(info_dict["trial_start"])
            self.trial_end = parse(info_dict["trial_end"])
            self.mission_start = parse(info_dict["mission_start"])
            self.mission_end = parse(info_dict["mission_end"])
            self.team_score = info_dict["team_score"]
            self.subject_id_to_color = info_dict["subject_id_to_color"]

        speeches_path = f"{save_directory}/speeches.pkl"
        if not os.path.exists(speeches_path):
            raise Exception(f"Could not find the file speeches.pkl in {save_directory}.")

        with open(speeches_path, "rb") as f:
            self.speeches_per_subject = pickle.load(f)

    def save(self, save_directory: str):
        info_dict = {
            "id": self.id,
            "number": self.number,
            "trial_start": self.trial_start.isoformat(),
            "trial_end": self.trial_end.isoformat(),
            "mission_start": self.mission_start.isoformat(),
            "mission_end": self.mission_end.isoformat(),
            "team_score": self.team_score,
            "subject_id_to_color": self.subject_id_to_color,
        }

        with open(f"{save_directory}/info.json", "w") as f:
            json.dump(info_dict, f, indent=4)

        with open(f"{save_directory}/speeches.pkl", "wb") as f:
            pickle.dump(self.speeches_per_subject, f)
