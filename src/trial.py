import json
from typing import Any, Dict, List
from datetime import datetime
import os


from dateutil.parser import parse
from src.components.speech.common import Utterance
from src.components.speech.vocalics_reader import VocalicsReader
from src.config.component_config_bundle import ComponentConfigBundle
from src.components.speech.vocalics_component import VocalicsComponent
from tqdm import tqdm


class Trial:
    """
    This class encapsulates information about an ASIST trial. It also contains a list of components we will use for
    coordination inference.
    """

    # For online inference, we have to change this to match on the header.type and msg.sub_type because the topic
    # won't be available as one of the message's field.
    TRIAL_TOPIC = "trial"
    ASR_TOPIC = "agent/asr/final"
    SCOREBOARD_TOPIC = "observations/events/scoreboard"
    MISSION_STATE_TOPIC = "observations/events/mission"
    VOCALIC_FEATURES = ["pitch", "intensity"]

    def __init__(self,
                 component_config_bundle: ComponentConfigBundle):

        self.component_config_bundle = component_config_bundle

        self.id = None
        self.number = None
        self.trial_start = None
        self.trial_end = None
        self.mission_start = None
        self.mission_end = None
        self.team_score = 0
        self.subject_id_map: Dict[str, str] = {}

        # Components
        self.vocalics = VocalicsComponent(component_config_bundle.vocalics_config)

    def save(self, out_dir: str):
        trial_out_dir = f"{out_dir}/{self.number}"
        os.makedirs(trial_out_dir, exist_ok=True)

        info_dict = {
            "id": self.id,
            "number": self.number,
            "trial_start": self.trial_start.isoformat(),
            "trial_end": self.trial_end.isoformat(),
            "mission_start": self.mission_start.isoformat(),
            "mission_end": self.mission_end.isoformat(),
            "team_score": self.team_score,
            "subject_id_map": self.subject_id_map,
        }

        self.component_config_bundle.save(trial_out_dir)
        self.vocalics.save(trial_out_dir)

        with open(f"{trial_out_dir}/info.json", "w") as f:
            json.dump(info_dict, f, indent=4)

    def load(self, trial_dir: str):
        info_path = f"{trial_dir}/info.json"
        if not os.path.exists(info_path):
            raise Exception(f"Could not find the file info.json in {trial_dir}.")

        with open(info_path, "r") as f:
            info_dict = json.load(f)
            self.id = info_dict["id"],
            self.number = info_dict["number"]
            self.trial_start = parse(info_dict["trial_start"])
            self.trial_end = parse(info_dict["trial_end"])
            self.mission_start = parse(info_dict["mission_start"])
            self.mission_end = parse(info_dict["mission_end"])
            self.team_score = info_dict["team_score"]
            self.subject_id_map = info_dict["subject_id_map"]

        self.component_config_bundle.load(trial_dir)
        self.vocalics.load(trial_dir)

        print("Ha")

    def parse(self, metadata_filepath: str):
        """
        Parses a metadata file and stores relevant information about the trial.
        """
        asr_messages = []

        with open(metadata_filepath, 'r') as f:
            for line in f:
                try:
                    json_message = json.loads(line)

                    topic = json_message.get("topic", "")
                    if topic == Trial.TRIAL_TOPIC:
                        self._parse_trial_info(json_message)
                    elif topic == Trial.MISSION_STATE_TOPIC:
                        self._parse_mission_state(json_message)
                    elif topic == Trial.SCOREBOARD_TOPIC:
                        self._parse_team_score(json_message)
                    elif topic == Trial.ASR_TOPIC:
                        asr_messages.append(json_message)
                except:
                    print(f"[ERROR] Bad json line of len: {len(line)}, {line}")

        time_range = (self.mission_start, self.mission_end)
        self.vocalics.parse(self.id, self.subject_id_map, asr_messages, time_range, self.trial_start)

    def _parse_trial_info(self, json_message: Any) -> None:
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
                subject_name = info["playername"]
                self.subject_id_map[subject_id] = subject_color
                self.subject_id_map[subject_name] = subject_color
        else:
            self.trial_end = parse(json_message["msg"]["timestamp"])

    def _parse_mission_state(self, json_message: Any) -> None:
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            self.mission_start = parse(json_message["header"]["timestamp"])
        else:
            self.mission_end = parse(json_message["header"]["timestamp"])

    def _parse_team_score(self, json_message: Any):
        self.team_score = max(int(json_message["data"]["scoreboard"]["TeamScore"]), self.team_score)
