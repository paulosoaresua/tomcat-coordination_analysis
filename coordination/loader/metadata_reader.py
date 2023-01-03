import os.path
from typing import Any, List, Tuple

from dateutil.parser import parse
import json
import logging

from coordination.entity.trial_metadata import Player, TrialMetadata
from coordination.entity.vocalics import Vocalics
from coordination.loader.vocalics_reader import VocalicsReader

logger = logging.getLogger()


class SurveyMappings:
    AGE = "QID1_TEXT"
    GENDER = "QID2"
    PROCESS_SCALE = ["QID801_1", "QID801_2", "QID801_3", "QID801_5", "QID801_6", "QID801_7", "QID801_8", "QID801_9",
                     "QID801_10"]
    TEAM_SATISFACTION = ["QID810_1", "QID810_2", "QID810_3", "QID750_1", "QID750_2"]
    NUMERICAL_GENDER_MAP = {
        1: "M",
        2: "F",
        3: "NB",
        4: "PNA",
    }


class MetadataReader:
    """
    This class is responsible for parsing a Minecraft .metadata file and extract relevant information about
    the trial for the coordination model.
    """

    TRIAL_TOPIC = "trial"
    ASR_TOPIC = "agent/asr/final"
    DIALOG_TOPIC = "agent/dialog"
    SCOREBOARD_TOPIC = "observations/events/scoreboard"
    MISSION_STATE_TOPIC = "observations/events/mission"
    SURVEY_RESPONSE_TOPIC = "status/asistdataingester/surveyresponse"

    def __init__(self, metadata_filepath: str, vocalics_reader: VocalicsReader):
        self._metadata_filepath = metadata_filepath
        self._vocalics_reader = vocalics_reader

    def load(self) -> Tuple[TrialMetadata, Vocalics]:
        trial_metadata = TrialMetadata()
        asr_messages = []
        dialog_messages = {}

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
                    elif topic == MetadataReader.SURVEY_RESPONSE_TOPIC:
                        self._parse_survey_response(json_message, trial_metadata)
                    elif topic == MetadataReader.ASR_TOPIC:
                        asr_messages.append(json_message)
                    elif topic == MetadataReader.DIALOG_TOPIC:
                        asr_id = json_message["data"]["asr_msg_id"]
                        dialog_messages[asr_id] = json_message
                except:
                    logger.error(f"Bad json line of len: {len(line)}, {line}")

        trial_metadata.check_validity()
        vocalics = Vocalics.from_asr_messages(asr_messages, trial_metadata, self._vocalics_reader, dialog_messages)

        # Remove player name from the entries in TrialMetadata.subject_id_map. We only add it so we can map both id
        # and name to avatar color because vocalics sometimes are indexed by one or the other.
        trial_metadata.subject_id_map = {key: value for i, (key, value) in
                                         enumerate(trial_metadata.subject_id_map.items()) if i % 2 == 0}

        return trial_metadata, vocalics

    @staticmethod
    def _parse_trial_info(json_message: Any, trial_metadata: TrialMetadata) -> None:
        if json_message["msg"]["sub_type"] == "start":
            trial_metadata.trial_start = parse(json_message["msg"]["timestamp"])
            trial_metadata.id = json_message["msg"]["trial_id"]
            trial_metadata.number = json_message["data"]["trial_number"]

            trial_metadata.subject_id_map = {}
            for info in json_message["data"]["client_info"]:
                avatar_color = info["callsign"].lower()
                subject_id = info["participant_id"]
                subject_name = info["playername"]
                player = Player(subject_id, subject_name, avatar_color)
                trial_metadata.subject_id_map[subject_id] = player
                trial_metadata.subject_id_map[subject_name] = player
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

    @staticmethod
    def _parse_survey_response(json_message: Any, trial_metadata: TrialMetadata):
        survey_name = (json_message["data"]["values"]["surveyname"]).lower()

        if "intakesurvey" in survey_name:
            MetadataReader._parse_demographic_survey(json_message, trial_metadata)
        elif "reflection" in survey_name:
            MetadataReader._parse_team_process_scale_survey(json_message, trial_metadata)
            MetadataReader._parse_team_satisfaction_survey(json_message, trial_metadata)

    @staticmethod
    def _parse_demographic_survey(json_message: Any, trial_metadata: TrialMetadata):
        player_id = json_message["data"]["values"]["participantid"]
        player = trial_metadata.subject_id_map[player_id]

        player.age = int(json_message["data"]["values"][SurveyMappings.AGE])
        numerical_gender = int(json_message["data"]["values"][SurveyMappings.GENDER])
        player.gender = SurveyMappings.NUMERICAL_GENDER_MAP.get(numerical_gender, "PNA")

    @staticmethod
    def _parse_team_process_scale_survey(json_message: Any, trial_metadata: TrialMetadata):
        player_id = json_message["data"]["values"]["participantid"]
        player = trial_metadata.subject_id_map[player_id]

        answers = [int(json_message["data"]["values"][question_id]) for question_id in SurveyMappings.PROCESS_SCALE]
        player.team_process_scale_survey_answers = answers

    @staticmethod
    def _parse_team_satisfaction_survey(json_message: Any, trial_metadata: TrialMetadata):
        player_id = json_message["data"]["values"]["participantid"]
        player = trial_metadata.subject_id_map[player_id]

        answers = [int(json_message["data"]["values"][question_id]) for question_id in SurveyMappings.TEAM_SATISFACTION]
        player.team_satisfaction_survey_answers = answers
