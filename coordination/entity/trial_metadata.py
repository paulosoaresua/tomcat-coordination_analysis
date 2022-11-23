from __future__ import annotations
from typing import Dict, List

from datetime import datetime
from dateutil.parser import parse
import json
import os

from coordination.common.utils import json_serial


class Player:
    id: str
    name: str
    avatar_color: str  # (red/green/blue)
    age: int
    gender: str  # (M - Male, F- Female, NB - NonBinary, PNA - Prefer Not to Answer)
    # Survey questions detailed in Appendix L of the pre-registration document in: https://osf.io/83vj2/
    team_process_scale_survey_answers: List[int]
    # Survey questions detailed in Appendix M of the pre-registration document in: https://osf.io/83vj2/
    team_satisfaction_survey_answers: List[int]

    def __init__(self, participant_id: str, participant_name: str, avatar_color: str):
        self.id = participant_id
        self.name = participant_name
        self.avatar_color = avatar_color


class TrialMetadata:
    id: str
    number: str
    trial_start: datetime
    trial_end: datetime
    mission_start: datetime
    mission_end: datetime
    team_score: int
    subject_id_map: Dict[str, Player]

    def __init__(self):
        self.team_score = 0

    @classmethod
    def from_trial_directory(cls, trial_dir: str) -> TrialMetadata:
        metadata_path = f"{trial_dir}/metadata.json"
        if not os.path.exists(metadata_path):
            raise Exception(f"Could not find the file info.json in {metadata_path}.")

        metadata = cls()
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
            for k, v in metadata_dict.items():
                if k in ["trial_start", "trial_end", "mission_start", "mission_end"]:
                    setattr(metadata, k, parse(v))
                else:
                    setattr(metadata, k, v)

        return metadata

    def check_validity(self):
        assert self.id
        assert self.number
        assert self.trial_start
        assert self.trial_end
        assert self.mission_start
        assert self.mission_end
        assert self.team_score
        assert self.subject_id_map

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump(self.__dict__, f, indent=4, default=json_serial)
