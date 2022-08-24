import os.path
from typing import Any, List, Tuple

from dateutil.parser import parse
import json
import logging

from coordination.entity.trial_metadata import TrialMetadata
from coordination.entity.vocalics import Vocalics
from coordination.config.database_config import DatabaseConfig

logger = logging.getLogger()


class MetadataReader:
    """
    This class is responsible for parsing a Minecraft .metadata file and extract relevant information about
    the trial for the coordination model.
    """

    TRIAL_TOPIC = "trial"
    ASR_TOPIC = "agent/asr/final"
    SCOREBOARD_TOPIC = "observations/events/scoreboard"
    MISSION_STATE_TOPIC = "observations/events/mission"

    def __init__(self, metadata_filepath: str, database_config: DatabaseConfig, vocalic_features: List[str]):
        self._metadata_filepath = metadata_filepath
        self._database_config = database_config
        self._vocalic_features = vocalic_features

    def load(self) -> Tuple[TrialMetadata, Vocalics]:
        trial_metadata = TrialMetadata()
        asr_messages = []

        logger.info(f"Parsing {os.path.basename(self._metadata_filepath)}")
        with open(self._metadata_filepath, 'r') as f:
            for line in f:
                try:
                    json_message = json.loads(line)

                    topic = json_message.get("topic", "")

                    # We match on the topic here because we are processing the metadata files offline.
                    # For online inference, we have to change this to match on the header.type and msg.sub_type
                    # because the topic won't be available as one of the message's field.
                    if topic == MetadataReader.TRIAL_TOPIC:
                        self._parse_trial_info(json_message, trial_metadata)
                    elif topic == MetadataReader.MISSION_STATE_TOPIC:
                        self._parse_mission_state(json_message, trial_metadata)
                    elif topic == MetadataReader.SCOREBOARD_TOPIC:
                        self._parse_team_score(json_message, trial_metadata)
                    elif topic == MetadataReader.ASR_TOPIC:
                        asr_messages.append(json_message)
                except:
                    logger.error(f"Bad json line of len: {len(line)}, {line}")

        trial_metadata.check_validity()
        vocalics = Vocalics.from_asr_messages(asr_messages, trial_metadata, self._database_config,
                                              self._vocalic_features)

        return trial_metadata, vocalics

    @staticmethod
    def _parse_trial_info(json_message: Any, trial_metadata: TrialMetadata) -> None:
        if json_message["msg"]["sub_type"] == "start":
            trial_metadata.trial_start = parse(json_message["msg"]["timestamp"])
            trial_metadata.id = json_message["msg"]["trial_id"]
            trial_metadata.number = json_message["data"]["trial_number"]

            trial_metadata.subject_id_map = {}
            for info in json_message["data"]["client_info"]:
                subject_callsign = info["callsign"].lower()
                subject_id = info["participant_id"]
                subject_name = info["playername"]
                trial_metadata.subject_id_map[subject_id] = subject_callsign
                trial_metadata.subject_id_map[subject_name] = subject_callsign
        else:
            trial_metadata.trial_end = parse(json_message["msg"]["timestamp"])

    @staticmethod
    def _parse_mission_state(json_message: Any, trial_metadata: TrialMetadata) -> None:
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            trial_metadata.mission_start = parse(json_message["header"]["timestamp"])
        else:
            trial_metadata.mission_end = parse(json_message["header"]["timestamp"])

    @staticmethod
    def _parse_team_score(json_message: Any, trial_metadata: TrialMetadata):
        score = int(json_message["data"]["scoreboard"]["TeamScore"])
        trial_metadata.team_score = max(score, trial_metadata.team_score)
